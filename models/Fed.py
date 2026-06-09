#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
import torch
import random
from torch import nn
from typing import Union
from .DMAB import MultiArmedBandit
import copy, torch
from typing import List, Dict, Tuple, Any
try:
    from models.dgc_integration import DGCManager
except Exception:
    from dgc_integration import DGCManager

def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result

def model_diff(w, w_init):
    diff = 0
    for k in w_init.keys():
        if not ("num_batches_tracked" in k):
            diff += torch.linalg.norm(w[k] - w_init[k])/(1 + torch.linalg.norm(w_init[k]))
    return diff
def model_diff2(w_new, w_old):
    total_diff = 0
    for key in w_old.keys():
        if 'num_batches_tracked' not in key:
            total_diff += torch.norm(w_new[key] - w_old[key]).item()
    return total_diff / (1 + total_diff)

def model_deviation(w_locals, w_init):
    model_deviation_list = []
    print("Num clients:",len(w_locals))
    for w in w_locals:
        model_deviation_list.append(model_diff(w, w_init).item())
    return model_deviation_list

class FedLearn(object):
    def __init__(self, args):
        self.args = args
    @staticmethod
    def FedAvg(w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg

    def FedSparseAggregateDGC(payloads: List[Dict[str, Tuple[Any, Any]]],
                              weights: List[float],
                              global_state: Dict[str, torch.Tensor],
                              dgc: DGCManager) -> Dict[str, torch.Tensor]:
        dense_delta = dgc.decompress_and_weighted_sum(payloads, weights, global_state)
        new_state = {}
        for name, param in global_state.items():
            delta = dense_delta.get(name)
            new_state[name] = param if delta is None else param + delta.to(param.device, dtype=param.dtype)
        return new_state


    def state_dict_difference(new_state: Dict[str, torch.Tensor],
                              base_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k in base_state.keys():
            out[k] = new_state[k] - base_state[k]
        return out


    def count_gradients(self, delta_w_locals, sparse_delta_w_locals):
         num_grads = []
         nz_grads = []
         for i in range(0, len(delta_w_locals)):
            num_grads.append(0)
            nz_grads.append(0)
         for k in delta_w_locals[0].keys():
            for i in range(len(delta_w_locals)):
                num_grads[i] += delta_w_locals[i][k].numel()
                nz_grads[i] += torch.nonzero(sparse_delta_w_locals[i][k]).size(0)
         return num_grads, nz_grads

class EdgeServer:
    def __init__(self, clients_idx, args):
        self.model = None
        self.delta = None
        self.fed_learner = FedLearn(args)
        self.clients_idx = sorted(clients_idx)
        self.args = args
        self.mab = MultiArmedBandit(
            num_arms=len(self.clients_idx),
            k=args.mab_k,
            delta=args.mab_delta,
            c=args.mab_c,
            q=args.mab_q
        )

        self.client_id_map = {i: cid for i, cid in enumerate(self.clients_idx)}
        self.global_to_local = {cid: idx for idx, cid in enumerate(self.clients_idx)}
        self.local_to_global = {idx: cid for idx, cid in enumerate(self.clients_idx)}

    def get_global_id(self, local_idx):
        """Map an edge-local client index to the global client id."""
        return self.local_to_global.get(local_idx, -1)

    def get_local_index(self, global_id):
        """Map a global client id to the edge-local index."""
        return self.global_to_local.get(global_id, -1)

    def MAB_selection(self, current_epoch):
        local_indices = self.mab.ucb(current_epoch=current_epoch)
        return [self.get_global_id(i) for i in local_indices]

    def MAB_selection2(self, current_epoch):
        local_indices = self.mab.ucb2(
            current_epoch=current_epoch,
            edge_server=self,
            edge_global_ids=self.clients_idx,
            select_k=getattr(self.args, "mab_k", self.mab.k),
            delta=getattr(self.args, "mab_delta", self.mab.delta),
            c=getattr(self.args, "mab_c", self.mab.c),
        )
        return [self.get_global_id(lid) for lid in local_indices]

    def gossip_selection(self):
        """Randomly select k clients from this edge server."""
        k = min(self.args.mab_k, len(self.clients_idx))
        rnd = random.Random(self.args.seed + int(getattr(self.args, "current_epoch", 0)) * 997 + id(self) % 7919)
        return rnd.sample(self.clients_idx, k)

    def greedy_selection(self):
        """Select the first k clients assigned to this edge server."""
        print("greedy")
        num_selected = min(self.args.mab_k, len(self.clients_idx))
        return self.clients_idx[:num_selected]
