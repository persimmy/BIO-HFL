import argparse
import copy
import threading
import torch
import torch.nn as nn

from models.Update import LocalUpdate
from models.Fed import FedLearn
from models.dgc_integration import DGCManager

def _unpack_comp(thing):
    if isinstance(thing, tuple) and len(thing) == 2 and isinstance(thing[1], dict):
        (values, idx), ctx = thing
        dense = bool(ctx.get('dense', False))
    else:
        values, idx = thing
        dense = False
    return values, idx, dense


def _weighted_agg_weights(num_samples_list):
    total = sum(num_samples_list)
    return [n/total for n in num_samples_list]


def _create_dgc_manager(args, global_model, uid: int):
    dgc_mgr = DGCManager(
        compress_ratio=args.dgc_ratio,
        warmup_epochs=args.dgc_warmup,
        fp16_values=args.dgc_fp16,
        res_decay=args.res_decay,
        res_clip_norm=(None if getattr(args, "res_clip_norm", None) in (None, 0) else args.res_clip_norm),
        enable_stats=args.dgc_stat_enable,
        stats_reset_every=args.dgc_stat_every,
        critical_tensor_enable=not bool(getattr(args, "dgc_disable_ct", False))
    )
    if hasattr(args, "result_dir"):
        setattr(dgc_mgr, "result_dir", args.result_dir)
    if getattr(args, "_dgc_names_dumped", False):
        setattr(dgc_mgr, "_dumped_names", True)
    dgc_mgr.initialize_from_model(global_model)
    setattr(args, "_dgc_names_dumped", True)
    setattr(dgc_mgr, "client_id", int(uid))
    return dgc_mgr


def _get_client_dgc_manager(args, global_model, uid: int):
    if not hasattr(args, "_dgc_mgr_by_client"):
        setattr(args, "_dgc_mgr_by_client", {})
    if not hasattr(args, "_dgc_mgr_lock"):
        setattr(args, "_dgc_mgr_lock", threading.Lock())

    managers = getattr(args, "_dgc_mgr_by_client")
    lock = getattr(args, "_dgc_mgr_lock")
    with lock:
        dgc_mgr = managers.get(int(uid))
        if dgc_mgr is None:
            print(f"[DGC] Create persistent DGCManager for client {uid}")
            dgc_mgr = _create_dgc_manager(args, global_model, int(uid))
            managers[int(uid)] = dgc_mgr
        return dgc_mgr

def train_one_round_with_dgc(args, global_model, selected_users, user_groups, dict_users_data):
    import time, traceback, gc
    from concurrent.futures import ThreadPoolExecutor, as_completed

    t0 = time.perf_counter()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f"[DGC] >>> Round begin | device={device} | selected_users={selected_users} | parallel_workers={getattr(args,'parallel_workers',4)}")

    tg = time.perf_counter()
    global_state = copy.deepcopy(global_model.state_dict())
    print(f"[DGC] Copied global_state in {time.perf_counter()-tg:.3f}s; #tensors={len(global_state)}")

    if args.dgc_enable:
        if not hasattr(args, "_dgc_mgr_by_client"):
            setattr(args, "_dgc_mgr_by_client", {})
        if not hasattr(args, "_dgc_mgr_lock"):
            setattr(args, "_dgc_mgr_lock", threading.Lock())
        existing = getattr(args, "_dgc_mgr_by_client")
        print(f"[DGC] Per-client DGC managers active | existing_clients={sorted(existing.keys())} | epoch={getattr(args, 'current_epoch', 'NA')}")



    payloads, local_states, num_samples = [], [], []
    per_client_log = {}

    def _train_one(uid: int):
        t_start = time.perf_counter()
        log = {
            'uid': uid,
            'idx_count': len(user_groups[uid]),
            'local_ep': getattr(args, 'local_ep', 1),
            'device': str(device),
            't_clone': 0.0,
            't_train': 0.0,
            't_delta': 0.0,
            't_compress': 0.0,
            'nz': 0,
            'payload_bytes': 0,
            'loss': None,
            'n_samp': 0,
            'error': None,
        }
        try:
            print(f"[DGC][Client {uid}] START | samples={log['idx_count']} | local_ep={log['local_ep']} | device={log['device']}")
            t_clone = time.perf_counter()
            local_model = copy.deepcopy(global_model).to(device, non_blocking=True)
            log['t_clone'] = time.perf_counter() - t_clone
            print(f"[DGC][Client {uid}] cloned+to({device}) in {log['t_clone']:.3f}s")

            local = LocalUpdate(
                args=args,
                dataset=dict_users_data[uid],
                idxs=user_groups[uid],
                local_ep=getattr(args, 'local_ep', 1),
                loader=getattr(args, 'loader_cache', {}).get(uid)
            )
            t_train = time.perf_counter()
            try:
                w_local, loss, n_samp = local.train(net=local_model)
            except ValueError:
                w_local, loss = local.train(net=local_model)
                n_samp = len(user_groups[uid])
            log['t_train'] = time.perf_counter() - t_train
            log['loss'] = float(loss)
            log['n_samp'] = int(n_samp)
            print(f"[DGC][Client {uid}] train() done in {log['t_train']:.3f}s | n_samp={n_samp} | loss={loss:.4f}")

            t_delta = time.perf_counter()

            param_keys = {n for (n, p) in global_model.named_parameters() if p.requires_grad}
            delta_params = {k: (w_local[k].detach().to('cpu', non_blocking=False) - global_state[k])
                            for k in param_keys}

            all_keys = set(global_state.keys())
            raw_buffer_keys = sorted(list(all_keys - param_keys))

            def _is_fp_tensor(t):
                return torch.is_floating_point(t)

            float_buffer_keys = [k for k in raw_buffer_keys if _is_fp_tensor(global_state[k])]
            delta_buffers = {k: (w_local[k].detach().to('cpu', non_blocking=False) - global_state[k])
                             for k in float_buffer_keys}

            log['t_delta'] = time.perf_counter() - t_delta

            if args.dgc_enable:
                t_compress = time.perf_counter()
                client_dgc_mgr = _get_client_dgc_manager(args, global_model, uid)
                if hasattr(args, "current_epoch"):
                    client_dgc_mgr.set_epoch(int(args.current_epoch))
                log['dgc_mgr_id'] = int(id(client_dgc_mgr))
                payload = client_dgc_mgr.compress_deltas(delta_params)

                # Buffers are model state (BN/BNTT running statistics), not CT.
                # Keep them synchronized even when CT is disabled; otherwise SNN
                # eval can collapse because BNTT running_mean/var stay stale.
                for k, tensor in delta_buffers.items():
                    flat = tensor.contiguous().view(-1)
                    values = (flat.to(dtype=torch.float16) if args.dgc_fp16 else flat.to(dtype=torch.float32))
                    idx = torch.arange(values.numel(), dtype=torch.int32)
                    ctx = dict(shape=tensor.shape, dtype=tensor.dtype, dense=True)
                    payload[k] = ((values, idx), ctx)

                log['t_compress'] = time.perf_counter() - t_compress

                try:
                    total_bytes = client_dgc_mgr.payload_nbytes(payload)
                    total_nz = 0
                    for name, thing in payload.items():
                        values, idx, dense = _unpack_comp(thing)
                        total_nz += int(values.numel()) if dense else int(idx.numel())
                    log['payload_bytes'] = int(total_bytes)
                    log['nz'] = int(total_nz)
                except Exception as e:
                    print(f"[DGC][Client {uid}] payload stats failed: {e}")
                    log['payload_bytes'] = 0
                    log['nz'] = 0

                print(f"[DGC][Client {uid}] delta in {log['t_delta']:.3f}s | compress in {log['t_compress']:.3f}s "
                      f"| payload~{log['payload_bytes']/1024:.1f}KB | nz={log['nz']} | mgr_id={log.get('dgc_mgr_id')}")

                del local_model
                if torch.cuda.is_available() and device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                gc.collect()

                return uid, ('payload', payload), n_samp, log
            else:
                print(f"[DGC][Client {uid}] no-compress path: returning full state_dict")
                del local_model
                if torch.cuda.is_available() and device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                gc.collect()
                return uid, ('state', w_local), n_samp, log

        except Exception as e:
            log['error'] = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            print(f"[DGC][Client {uid}] ERROR:\n{log['error']}")
            try:
                if torch.cuda.is_available() and device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
            except:
                pass
            gc.collect()
            return uid, None, 0, log

    max_workers = 1 if device.type == 'cuda' else getattr(args, "parallel_workers", 4)
    print(f"[DGC] Submitting {len(selected_users)} clients to ThreadPoolExecutor(max_workers={max_workers}) ...")
    t_pool = time.perf_counter()
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_train_one, uid) for uid in selected_users]
        for fut in as_completed(futures):
            uid, result, n_samp, log = fut.result()
            if result is None:
                failed += 1
                print(f"[DGC] Client {uid} failed and is skipped.")
                continue
            kind, payload_or_state = result
            per_client_log[uid] = log
            num_samples.append(int(n_samp))
            if kind == 'payload':
                payloads.append(payload_or_state)
            else:
                local_states.append(payload_or_state)

    print(f"[DGC] ThreadPool finished in {time.perf_counter()-t_pool:.3f}s | finished={len(selected_users)-failed}/{len(selected_users)} | failed={failed}")

    t_agg = time.perf_counter()
    if args.dgc_enable:
        total = sum(num_samples) if num_samples else 1
        weights = [n / total for n in num_samples]
        print(f"[DGC] Aggregating sparse payloads | num_clients={len(payloads)} | total_samples={total}")
        managers = getattr(args, "_dgc_mgr_by_client", {})
        agg_dgc_mgr = next(iter(managers.values()), None)
        if agg_dgc_mgr is None:
            agg_dgc_mgr = _create_dgc_manager(args, global_model, uid=-1)
        new_state = FedLearn.FedSparseAggregateDGC(payloads, weights, global_state, agg_dgc_mgr)

        try:
            total_upload = sum(agg_dgc_mgr.payload_nbytes(p) for p in payloads)
        except Exception:
            total_upload = 0
        avg_upload = int(total_upload / max(1, len(payloads)))
        per_client_bytes = {int(uid): int(lg.get('payload_bytes', 0))
                            for uid, lg in per_client_log.items()}
        per_client_loss = {int(uid): float(lg.get('loss'))
                           for uid, lg in per_client_log.items()
                           if lg.get('loss') is not None}
        per_client_train_time = {int(uid): float(lg.get('t_train', 0.0))
                                 for uid, lg in per_client_log.items()}
        per_client_samples = {int(uid): int(lg.get('n_samp', 0))
                              for uid, lg in per_client_log.items()}

        dgc_stats = {
            'avg_upload_bytes': int(avg_upload),
            'total_upload_bytes': int(total_upload),
            'per_client_bytes': per_client_bytes,
            'per_client_loss': per_client_loss,
            'per_client_train_time': per_client_train_time,
            'per_client_samples': per_client_samples,
            'per_client_dgc_manager_ids': {
                int(uid): int(lg.get('dgc_mgr_id', 0))
                for uid, lg in per_client_log.items()
            },
        }

        print(f"[DGC] Aggregate done in {time.perf_counter()-t_agg:.3f}s | avg_upload~{avg_upload/1024:.1f}KB | total~{total_upload/(1024*1024):.2f}MB")
        # Report/reset DGC stats for the client-specific residual memories used this round.
        managers = getattr(args, "_dgc_mgr_by_client", {})
        for uid in sorted(per_client_log.keys()):
            mgr = managers.get(int(uid))
            if mgr is not None and getattr(mgr, "end_round", None):
                mgr.end_round()
        print(f"[DGC] >>> Round end in {time.perf_counter()-t0:.3f}s")
        for uid in sorted(per_client_log.keys()):
            lg = per_client_log[uid]
            status = "OK" if lg['error'] is None else "FAIL"
            print(f"[DGC][Summary][Client {uid}] {status} | clone {lg['t_clone']:.2f}s | train {lg['t_train']:.2f}s | "
                  f"delta {lg['t_delta']:.2f}s | comp {lg['t_compress']:.2f}s | payload {lg['payload_bytes']/1024:.1f}KB | nz={lg['nz']} | idx={lg['idx_count']}")
        return new_state, dgc_stats
    else:
        print(f"[DGC] Aggregating dense states | num_clients={len(local_states)}")
        w_glob = FedLearn.FedAvg(local_states)
        print(f"[DGC] Aggregate done in {time.perf_counter()-t_agg:.3f}s")
        print(f"[DGC] >>> Round end in {time.perf_counter()-t0:.3f}s")
        for uid in sorted(per_client_log.keys()):
            lg = per_client_log[uid]
            status = "OK" if lg['error'] is None else "FAIL"
            print(f"[DGC][Summary][Client {uid}] {status} | clone {lg['t_clone']:.2f}s | train {lg['t_train']:.2f}s | "
                  f"idx={lg['idx_count']}")
        dense_stats = {
            'avg_upload_bytes': 0,
            'total_upload_bytes': 0,
            'per_client_bytes': {},
            'per_client_loss': {
                int(uid): float(lg.get('loss'))
                for uid, lg in per_client_log.items()
                if lg.get('loss') is not None
            },
            'per_client_train_time': {
                int(uid): float(lg.get('t_train', 0.0))
                for uid, lg in per_client_log.items()
            },
            'per_client_samples': {
                int(uid): int(lg.get('n_samp', 0))
                for uid, lg in per_client_log.items()
            },
        }
        return w_glob, dense_stats
