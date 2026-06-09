#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader, Subset
import math
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from utils.sampling import cifar_iid, cifar_non_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Fed import FedLearn
from models.Fed import EdgeServer
from models.Fed import model_diff2
from models.test import test_img, test_img_metrics
import models.vgg_spiking_bntt as snn_models_bntt
from models.main_fed_dgc_stub import train_one_round_with_dgc
from models.initialenergy import (
    compute_energy_snn,
    comm_energy_from_bytes, typical_client_distance_m, state_dict_size_bytes
)
from client_selection import ClientSelectionManager
from utils.fairness import edge_average_jain, jain_index
import os
import time

import glob
import json
import sys

from utils.dagm_dataset import (
    DAGM2007MultiClassDataset,
    build_dagm_transforms,
    stratified_iid_partition,
    dirichlet_noniid_partition,
)

class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def to_cpu_state_dict(sd: dict):
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}


def build_client_distance_profile(args):
    profile = str(getattr(args, 'client_distance_profile', 'default')).lower()
    if profile == 'default':
        return {cid: typical_client_distance_m() for cid in range(args.num_users)}

    lo = float(getattr(args, 'client_distance_min', 30.0))
    hi = float(getattr(args, 'client_distance_max', 800.0))
    if hi < lo:
        lo, hi = hi, lo
    lo = max(1.0, lo)
    hi = max(lo, hi)

    distance_seed = getattr(args, 'client_distance_seed', None)
    rnd_seed = int(distance_seed) if distance_seed is not None else int(args.seed) + 20240528
    rnd = random.Random(rnd_seed)
    if profile == 'extreme':
        distances = [float(x) for x in np.geomspace(lo, hi, int(args.num_users))]
        rnd.shuffle(distances)
    elif profile == 'loguniform':
        log_lo, log_hi = math.log(lo), math.log(hi)
        distances = [float(math.exp(rnd.uniform(log_lo, log_hi))) for _ in range(args.num_users)]
    else:
        raise ValueError(f"Unknown client_distance_profile: {profile}")

    return {cid: distances[cid] for cid in range(args.num_users)}


def scheduled_learning_rate(args, global_round):
    base_lr = float(getattr(args, 'base_lr', getattr(args, 'lr', 0.0)))
    warmdown_rounds = int(getattr(args, 'lr_warmdown_rounds', 0) or 0)
    target_lr = float(getattr(args, 'lr_warmdown_target', 0.0) or 0.0)
    if warmdown_rounds <= 0 or target_lr <= 0.0:
        return base_lr
    if warmdown_rounds == 1:
        return target_lr
    if global_round >= warmdown_rounds - 1:
        return target_lr
    progress = float(global_round) / float(warmdown_rounds - 1)
    return base_lr + (target_lr - base_lr) * progress


if __name__ == '__main__':
    # parse args
    args = args_parser()
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    _log_file = open(Path(args.result_dir) / 'run.log', 'w', encoding='utf-8', buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, _log_file)
    sys.stderr = TeeStream(sys.__stderr__, _log_file)
    print("Program start")
    args.parallel_workers = getattr(args, "parallel_workers", 1)
    args.num_workers = int(os.getenv("NUM_WORKERS", getattr(args, "num_workers", 0)))
    args.base_lr = float(args.lr)
    random.seed(args.seed)
    client_distance_m = build_client_distance_profile(args)
    print(f"Client distance profile: {args.client_distance_profile}")
    print("Client distances (m): " + json.dumps({int(k): round(float(v), 3) for k, v in client_distance_m.items()}, sort_keys=True))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    all_clients = [{
        'global_id': i,
        'local_ep': int(getattr(args, 'local_ep', 1))
    } for i in range(args.num_users)]
    np.random.shuffle(all_clients)
    client_dict = {c['global_id']: c['local_ep'] for c in all_clients}
    max_id = max(client_dict.keys()) if client_dict else 0
    local_ep_array = np.array([client_dict.get(i, 0) for i in range(max_id + 1)])
    edge_users = []
    clients_per_edge = args.num_users // args.num_edges

    for i in range(args.num_edges):
        start = i * clients_per_edge
        end = (i + 1) * clients_per_edge if i != args.num_edges - 1 else args.num_users
        edge_users.append(all_clients[start:end])
    print("\n=== Edge Server Client Distribution ===")
    for i, edge_clients in enumerate(edge_users):
        print(f"Edge {i} has {len(edge_clients)} clients: {[c['global_id'] for c in edge_clients]}")
    #print(edge_users)

    edge_servers = [
        EdgeServer(
            clients_idx=[client['global_id'] for client in edge_users[u]],
            args=args
        ) for u in range(args.num_edges)
    ]
    selection_manager = ClientSelectionManager(args)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    print(f"Client selection strategy: {selection_manager.method}")
    #print(edge_servers)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.device != 'cpu':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset_keys = None
    h5fs = None
    # load dataset and split users
    if args.dataset == 'CIFAR10':
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2470, 0.2430, 0.2610)

        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

        dataset_train = datasets.CIFAR10('data/cifar', train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10('data/cifar', train=False, download=True, transform=trans_test)

        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_non_iid(dataset_train, args.num_classes, args.num_users, alpha=args.dirichlet_alpha)
    elif args.dataset == 'DAGM2007':
        dagm_transform = build_dagm_transforms(img_size=32)
        dataset_train = DAGM2007MultiClassDataset(args.dagm_data_dir, split='train', transform=dagm_transform)
        dataset_test = DAGM2007MultiClassDataset(args.dagm_data_dir, split='test', transform=dagm_transform)
        args.num_classes = int(len(dataset_train.classes))
        args.input_channels = 1
        print(f"DAGM2007 loaded from {args.dagm_data_dir}")
        print(f"DAGM2007 classes: {dataset_train.classes}")
        print(f"DAGM2007 train/test samples: {len(dataset_train)}/{len(dataset_test)}")
        if args.iid:
            dict_users = stratified_iid_partition(dataset_train.targets, args.num_users, args.seed)
            print("DAGM2007 partition: stratified IID")
        else:
            dict_users = dirichlet_noniid_partition(
                dataset_train.targets,
                args.num_users,
                alpha=args.dirichlet_alpha,
                seed=args.seed,
            )
            print(f"DAGM2007 partition: Dirichlet non-IID alpha={args.dirichlet_alpha}")
    else:
        exit('Error: this public release supports only CIFAR10 and DAGM2007')
    # img_size = dataset_train[0][0].shape
    dict_users_data = {i: dataset_train for i in range(args.num_users)}


    def build_loader_cache(dataset, dict_users, args):
        nw = min(getattr(args, "num_workers", 0), 2)
        loader_cache = {}
        for cid in range(args.num_users):
            idxs = list(dict_users[cid])  # set -> list
            subset = Subset(dataset, idxs)
            loader_cache[cid] = DataLoader(
                subset,
                batch_size=args.local_bs,
                shuffle=True,
                num_workers=nw,
                pin_memory=True,
                persistent_workers=True if nw > 0 else False,
                prefetch_factor=2 if nw > 0 else None,
                drop_last=False
            )
        return loader_cache


    args.loader_cache = build_loader_cache(dataset_train, dict_users, args)
    # ============================================================================
    # build model
    if args.model[0:3].lower() == 'vgg':
        model_args = {
            'num_cls': args.num_classes,
            'timesteps': args.timesteps,
            'input_channels': int(getattr(args, 'input_channels', 3)),
        }
        net_glob = snn_models_bntt.SNN_VGG9_BNTT(**model_args).cuda()
    else:
        exit('Error: unrecognized model')


    net_glob = net_glob.to(args.device)

    mean_set=np.zeros(args.num_users)
    energy_mean_set=np.zeros(args.num_users)
    epoch_mean_set=np.zeros(args.num_users)
    total_client_snn_energy = np.zeros(args.num_users)
    total_client_energy = np.zeros(args.num_users)
    current_global_client_energy = np.zeros(args.num_users)

    total_global_energy = [0.0] * args.epochs
    total_global_snn_energy = [0.0] * args.epochs
    loss_train_list = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    rewards_list = []
    selected_global_ids = []

    # metrics to store
    ms_acc_train_list, ms_loss_train_list = [], []
    ms_acc_test_list, ms_loss_test_list = [], []
    ms_num_client_list, ms_tot_comm_cost_list, ms_avg_comm_cost_list, ms_max_comm_cost_list = [], [], [], []
    ms_tot_nz_grad_list, ms_avg_nz_grad_list, ms_max_nz_grad_list = [], [], []
    ms_model_deviation = []
    ms_global_time_list = []
    acc_train, loss_train = 0, 0
    acc_test, loss_test = 0, 0
    macro_f1_test = 0.0
    per_class_f1_test = []
    per_class_precision_test = []
    per_class_recall_test = []
    confusion_matrix_test = []
    selection_counts = np.zeros(args.num_users, dtype=int)
    total_upload_bytes = 0
    total_download_bytes = 0
    detailed_metrics_rows = []

    _t0_global = time.perf_counter()
    for global_round in range(args.epochs):
        args.current_epoch = global_round
        args.current_lr = scheduled_learning_rate(args, global_round)
        args.mab_delta = 1 / (global_round + 1)
        print("\n=========== Global Round [%d/%d] ===========" % (global_round + 1, args.epochs))
        print("Local client learning rate: {:.6g}".format(float(args.current_lr)))
        global_init = to_cpu_state_dict(net_glob.state_dict())
        edge_deltas = []
        selected_by_edge_round = {}
        round_train_losses = []
        round_client_loss_lists = {}
        round_upload_bytes = 0
        round_download_bytes = 0


        for edge_idx, edge in enumerate(edge_servers):

            edge.edge_id = edge_idx
            print(f"\n-- EDGE SERVER {edge_idx + 1}/{len(edge_servers)} --")

            #selected_clients = edge.MAB_selection2(current_epoch=global_round + 1)

            selected_clients = selection_manager.select(edge, global_round, selection_counts)
            selected_by_edge_round[int(edge_idx)] = [int(cid) for cid in selected_clients]
            for cid in selected_clients:
                selection_counts[int(cid)] += 1

            #selected_clients = edge.greedy_selection()
            print(f"Selected Clients: {selected_clients}")

            edge_model = copy.deepcopy(global_init)

            round_energy_acc = {gid: 0.0 for gid in edge.clients_idx}

            for edge_epoch in range(args.edge_rounds):
                print(f"Edge Round {edge_epoch + 1} with clients {selected_clients}")

                temp_edge_module = copy.deepcopy(net_glob).cpu()
                temp_edge_module.load_state_dict(edge_model, strict=True)

                new_state, dgc_stats = train_one_round_with_dgc(
                    args=args,
                    global_model=temp_edge_module,
                    selected_users=selected_clients,
                    user_groups=dict_users,
                    dict_users_data=dict_users_data
                )

                edge_model = to_cpu_state_dict(new_state)
                per_client_loss = (dgc_stats or {}).get('per_client_loss', {})
                for cid, loss_val in per_client_loss.items():
                    cid = int(cid)
                    loss_val = float(loss_val)
                    round_client_loss_lists.setdefault(cid, []).append(loss_val)
                    round_train_losses.append(loss_val)

                bytes_map = {}
                if args.dgc_enable and dgc_stats:
                    bytes_map = dgc_stats.get('per_client_bytes', {}) or {}
                    avg_upload_bytes = int(dgc_stats.get('avg_upload_bytes', 0))
                else:
                    avg_upload_bytes = 0

                downlink_bytes_full_model = state_dict_size_bytes(edge_model)

                for cid in selected_clients:
                    distance_m = client_distance_m[cid]
                    if bytes_map:
                        uplink_bytes = int(bytes_map.get(cid, avg_upload_bytes))
                    else:
                        uplink_bytes = downlink_bytes_full_model

                    round_upload_bytes += int(uplink_bytes)
                    round_download_bytes += int(downlink_bytes_full_model)
                    uplink_E = comm_energy_from_bytes(uplink_bytes, distance_m, args)
                    comm_E = uplink_E

                    samples = int(len(dict_users[cid]) * int(getattr(args, 'local_ep', 1)))
                    comp_E = compute_energy_snn(samples=samples, layer_rates=None, args=args)
                    total_client_snn_energy[cid] += comp_E

                    total_client_energy[cid] += (comp_E + comm_E)
                    round_energy_acc[cid] += (comp_E + comm_E)


            edge_loss_update = {}
            for cid in selected_clients:
                vals = round_client_loss_lists.get(int(cid), [])
                if vals:
                    edge_loss_update[int(cid)] = float(sum(vals) / len(vals))
            selection_manager.update_losses(edge_loss_update)

            E_vals_sel = [float(round_energy_acc[gid]) for gid in selected_clients]
            E_max = max(E_vals_sel) if len(E_vals_sel) > 0 else 0.0
            if E_max <= 1e-12:
                energy_mean_set = {gid: 0.0 for gid in selected_clients}
            else:
                energy_mean_set = {gid: 1.0 - (float(round_energy_acc[gid]) / E_max) for gid in selected_clients}

            for gid in selected_clients:
                lid = edge.get_local_index(gid)
                e_hat = float(energy_mean_set[gid])
                edge.mab.accumulate_energy(lid, e_hat)

            t_completed = int(global_round + 1)
            rewards_all = edge.mab.build_rewards_for_all(
                edge_server=edge,
                t_completed=t_completed,
                selected_set=set(selected_clients),
                q=edge.mab.q
            )

            edge.mab.update(
                edge_server=edge,
                selected_clients=selected_clients,
                rewards=rewards_all
            )

            delta = {}
            for key in edge_model:
                delta[key] = (edge_model[key] - global_init[key]) * args.lambda_weight
            edge_deltas.append(delta)

        w_avg = copy.deepcopy(global_init)
        for key in w_avg.keys():
            total_delta = torch.zeros_like(w_avg[key], dtype=torch.float)

            for delta in edge_deltas:
                total_delta += delta[key].to(torch.float)

            w_avg[key] += (total_delta / len(edge_servers)).to(w_avg[key].dtype)

        net_glob.load_state_dict(w_avg, strict=True)

        if global_round % args.eval_every == 0:
            net_glob.eval()
            eval_metrics = test_img_metrics(net_glob, dataset_test, args)
            acc_test = eval_metrics["acc"]
            loss_test = eval_metrics["loss"]
            macro_f1_test = eval_metrics["macro_f1"]
            per_class_f1_test = eval_metrics["per_class_f1"]
            per_class_precision_test = eval_metrics["per_class_precision"]
            per_class_recall_test = eval_metrics["per_class_recall"]
            confusion_matrix_test = eval_metrics["confusion_matrix"]
            print("Global Round {:d}, Testing accuracy: {:.2f}%, Test loss: {:.2f}, Macro-F1: {:.4f}".format(
                global_round, acc_test, loss_test, macro_f1_test
            ))

            ms_acc_test_list.append(acc_test)
            ms_loss_test_list.append(loss_test)

        round_snn_energy = sum(total_client_snn_energy) - (total_global_snn_energy[global_round - 1] if global_round > 0 else 0)
        total_global_snn_energy[global_round] = round_snn_energy + (
            total_global_snn_energy[global_round - 1] if global_round > 0 else 0)

        round_energy = sum(total_client_energy) - (total_global_energy[global_round - 1] if global_round > 0 else 0)
        total_global_energy[global_round] = round_energy + (
            total_global_energy[global_round - 1] if global_round > 0 else 0)
        print(f"\n[Energy Report] Global Round {global_round + 1}:")
        print(f"Current Round Energy: {round_energy:.2f} J")
        print(f"Accumulated Energy: {total_global_energy[global_round]:.2f} J")

        global_time_s = time.perf_counter() - _t0_global
        ms_global_time_list.append(global_time_s)
        print(f"Global time: {ms_global_time_list[global_round]:.2f} s")

        total_upload_bytes += int(round_upload_bytes)
        total_download_bytes += int(round_download_bytes)
        client_loss_avg = {
            int(cid): float(sum(vals) / len(vals))
            for cid, vals in round_client_loss_lists.items()
            if vals
        }
        jain_global = jain_index(selection_counts)
        jain_edge_avg = edge_average_jain(selection_counts, edge_servers)
        detailed_metrics_rows.append({
            'round': int(global_round + 1),
            'algorithm': selection_manager.method,
            'seed': int(args.seed),
            'dataset': args.dataset,
            'iid': bool(args.iid),
            'dirichlet_alpha': float(getattr(args, 'dirichlet_alpha', 0.5)),
            'timesteps': int(args.timesteps),
            'lr': float(getattr(args, 'current_lr', args.lr)),
            'base_lr': float(getattr(args, 'base_lr', args.lr)),
            'lr_warmdown_rounds': int(getattr(args, 'lr_warmdown_rounds', 0) or 0),
            'lr_warmdown_target': float(getattr(args, 'lr_warmdown_target', 0.0) or 0.0),
            'client_distance_profile': str(getattr(args, 'client_distance_profile', 'default')),
            'client_distance_seed': getattr(args, 'client_distance_seed', None),
            'client_distance_m': json.dumps({int(k): float(v) for k, v in client_distance_m.items()}, sort_keys=True),
            'dgc_enable': bool(args.dgc_enable),
            'dgc_ratio': float(getattr(args, 'dgc_ratio', 1.0)),
            'dgc_warmup': int(getattr(args, 'dgc_warmup', 0)),
            'dgc_fp16': bool(getattr(args, 'dgc_fp16', False)),
            'dgc_disable_ct': bool(getattr(args, 'dgc_disable_ct', False)),
            'ct_enable_implicit': bool(args.dgc_enable and not getattr(args, 'dgc_disable_ct', False)),
            'test_acc': float(acc_test),
            'test_loss': float(loss_test),
            'test_macro_f1': float(macro_f1_test),
            'per_class_f1': json.dumps(per_class_f1_test),
            'per_class_precision': json.dumps(per_class_precision_test),
            'per_class_recall': json.dumps(per_class_recall_test),
            'confusion_matrix': json.dumps(confusion_matrix_test),
            'round_train_loss_avg': float(sum(round_train_losses) / len(round_train_losses)) if round_train_losses else None,
            'selected_clients_by_edge': json.dumps(selected_by_edge_round, sort_keys=True),
            'client_select_counts': json.dumps([int(x) for x in selection_counts.tolist()]),
            'client_train_loss': json.dumps(client_loss_avg, sort_keys=True),
            'oort_loss_scores': json.dumps(selection_manager.latest_loss, sort_keys=True),
            'jain_selection_global': float(jain_global),
            'jain_selection_edge_avg': float(jain_edge_avg),
            'round_energy': float(round_energy),
            'total_energy': float(total_global_energy[global_round]),
            'round_snn_energy': float(round_snn_energy),
            'total_snn_energy': float(total_global_snn_energy[global_round]),
            'round_upload_bytes': int(round_upload_bytes),
            'round_download_bytes': int(round_download_bytes),
            'total_upload_bytes': int(total_upload_bytes),
            'total_download_bytes': int(total_download_bytes),
            'global_time_s': float(global_time_s),
            'mab_q': float(getattr(args, 'mab_q', 0.0)),
        })


    detailed_df = pd.DataFrame(detailed_metrics_rows)
    detailed_df.to_csv(Path(args.result_dir) / 'selection_metrics.csv', sep='\t', index=False)

    # Keep the old curve CSV shape, but derive it from the detailed per-round log.
    metrics_df = pd.DataFrame({
        'Test acc': detailed_df['test_acc'],
        'Test loss': detailed_df['test_loss'],
        'Macro F1': detailed_df['test_macro_f1'],
        'Total energy': detailed_df['total_energy'],
        'Total snn energy': detailed_df['total_snn_energy'],
        'Global time (s)': detailed_df['global_time_s'],
    })
    metrics_df.to_csv(Path(args.result_dir) / 'try.csv', sep='\t', index=False)

    def _first_reach(threshold):
        reached = detailed_df[detailed_df['test_acc'] >= threshold]
        if reached.empty:
            return None, None, None
        row = reached.iloc[0]
        return int(row['round']), float(row['total_energy']), float(row['global_time_s'])

    r60, e60, t60 = _first_reach(60.0)
    r65, e65, t65 = _first_reach(65.0)

    def _first_reach_f1(threshold):
        reached = detailed_df[detailed_df['test_macro_f1'] >= threshold]
        if reached.empty:
            return None, None, None
        row = reached.iloc[0]
        return int(row['round']), float(row['total_energy']), float(row['global_time_s'])

    rf80, ef80, tf80 = _first_reach_f1(0.80)
    rf90, ef90, tf90 = _first_reach_f1(0.90)
    summary_df = pd.DataFrame([{
        'algorithm': selection_manager.method,
        'seed': int(args.seed),
        'dataset': args.dataset,
        'iid': bool(args.iid),
        'dirichlet_alpha': float(getattr(args, 'dirichlet_alpha', 0.5)),
        'best_acc': float(detailed_df['test_acc'].max()),
        'final_acc': float(detailed_df['test_acc'].iloc[-1]),
        'best_macro_f1': float(detailed_df['test_macro_f1'].max()),
        'final_macro_f1': float(detailed_df['test_macro_f1'].iloc[-1]),
        'round_to_60_acc': r60,
        'energy_to_60_acc': e60,
        'time_to_60_acc': t60,
        'round_to_65_acc': r65,
        'energy_to_65_acc': e65,
        'time_to_65_acc': t65,
        'round_to_0p80_macro_f1': rf80,
        'energy_to_0p80_macro_f1': ef80,
        'time_to_0p80_macro_f1': tf80,
        'round_to_0p90_macro_f1': rf90,
        'energy_to_0p90_macro_f1': ef90,
        'time_to_0p90_macro_f1': tf90,
        'auc_acc_round': float(np.trapz(detailed_df['test_acc'], detailed_df['round'])),
        'auc_acc_energy': float(np.trapz(detailed_df['test_acc'], detailed_df['total_energy'])),
        'auc_macro_f1_round': float(np.trapz(detailed_df['test_macro_f1'], detailed_df['round'])),
        'auc_macro_f1_energy': float(np.trapz(detailed_df['test_macro_f1'], detailed_df['total_energy'])),
        'final_jain_selection_global': float(detailed_df['jain_selection_global'].iloc[-1]),
        'final_jain_selection_edge_avg': float(detailed_df['jain_selection_edge_avg'].iloc[-1]),
    }])
    summary_df.to_csv(Path(args.result_dir) / 'summary.csv', sep='\t', index=False)
