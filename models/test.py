#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import contextlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

@contextlib.contextmanager
def temp_attr(obj, name, value):
    """Temporarily override one model attribute inside a context manager."""
    has = hasattr(obj, name)
    old = getattr(obj, name, None)
    if has:
        setattr(obj, name, value)
    try:
        yield
    finally:
        if has:
            setattr(obj, name, old)


def test_img(net_g, datatest, args):
    """Evaluate accuracy and loss with optional mixed precision and timestep override."""
    device = args.device
    amp_eval = bool(getattr(args, 'amp_eval', True) and device.type == 'cuda')
    eval_bs = int(getattr(args, 'eval_bs', max(args.bs, 64)))
    eval_max_batches = int(getattr(args, 'eval_max_batches', 0))
    eval_timesteps = int(getattr(args, 'eval_timesteps', 0))

    nw = min(int(getattr(args, 'num_workers', 4)), 2)  # <= 2
    loader = DataLoader(
        datatest,
        batch_size=eval_bs,
        shuffle=False,
        pin_memory=(device.type == 'cuda'),
        num_workers=nw,
        persistent_workers=False,
    )

    model = net_g.module if isinstance(net_g, torch.nn.DataParallel) and torch.cuda.device_count() == 1 else net_g
    model.eval()

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    total_loss = 0.0
    total_correct = 0
    seen = 0

    ctx = (
        temp_attr(model, 'timesteps', eval_timesteps)
        if (eval_timesteps > 0 and hasattr(model, 'timesteps'))
        else contextlib.nullcontext()
    )
    with ctx, torch.inference_mode():
        if amp_eval:
            if torch.cuda.is_available():
                scaler_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                scaler_ctx = torch.amp.autocast(device_type="mps", dtype=torch.float16)
            else:
                scaler_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
        else:
            scaler_ctx = contextlib.nullcontext()
        with scaler_ctx:
            for b, (data, target) in enumerate(loader, 1):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                logits = model(data)
                loss = F.cross_entropy(logits, target, reduction='sum')
                pred = logits.argmax(dim=1)
                correct = (pred == target).sum().item()

                total_loss += loss.item()
                total_correct += correct
                seen += target.size(0)

                if eval_max_batches and b >= eval_max_batches:
                    break

    if seen == 0:
        return 0.0, float('inf')
    return 100.0 * total_correct / seen, total_loss / seen


def _classification_metrics(targets, preds, num_classes):
    targets = np.asarray(targets, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    per_precision, per_recall, per_f1 = [], [], []
    for cls in range(num_classes):
        tp = float(cm[cls, cls])
        fp = float(cm[:, cls].sum() - cm[cls, cls])
        fn = float(cm[cls, :].sum() - cm[cls, cls])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_precision.append(precision)
        per_recall.append(recall)
        per_f1.append(f1)

    return {
        "macro_f1": float(np.mean(per_f1)) if per_f1 else 0.0,
        "per_class_precision": per_precision,
        "per_class_recall": per_recall,
        "per_class_f1": per_f1,
        "confusion_matrix": cm.tolist(),
    }


def test_img_metrics(net_g, datatest, args):
    device = args.device
    amp_eval = bool(getattr(args, 'amp_eval', True) and device.type == 'cuda')
    eval_bs = int(getattr(args, 'eval_bs', max(args.bs, 64)))
    eval_max_batches = int(getattr(args, 'eval_max_batches', 0))
    eval_timesteps = int(getattr(args, 'eval_timesteps', 0))
    num_classes = int(getattr(args, 'num_classes', 10))

    nw = min(int(getattr(args, 'num_workers', 4)), 2)
    loader = DataLoader(
        datatest,
        batch_size=eval_bs,
        shuffle=False,
        pin_memory=(device.type == 'cuda'),
        num_workers=nw,
        persistent_workers=False,
    )

    model = net_g.module if isinstance(net_g, torch.nn.DataParallel) and torch.cuda.device_count() == 1 else net_g
    model.eval()

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    total_loss = 0.0
    total_correct = 0
    seen = 0
    all_targets, all_preds = [], []

    ctx = (
        temp_attr(model, 'timesteps', eval_timesteps)
        if (eval_timesteps > 0 and hasattr(model, 'timesteps'))
        else contextlib.nullcontext()
    )
    with ctx, torch.inference_mode():
        if amp_eval:
            if torch.cuda.is_available():
                scaler_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                scaler_ctx = torch.amp.autocast(device_type="mps", dtype=torch.float16)
            else:
                scaler_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
        else:
            scaler_ctx = contextlib.nullcontext()
        with scaler_ctx:
            for b, (data, target) in enumerate(loader, 1):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                logits = model(data)
                loss = F.cross_entropy(logits, target, reduction='sum')
                pred = logits.argmax(dim=1)

                total_loss += loss.item()
                total_correct += (pred == target).sum().item()
                seen += target.size(0)
                all_targets.extend(target.detach().cpu().tolist())
                all_preds.extend(pred.detach().cpu().tolist())

                if eval_max_batches and b >= eval_max_batches:
                    break

    if seen == 0:
        out = _classification_metrics([], [], num_classes)
        out.update({"acc": 0.0, "loss": float('inf')})
        return out

    out = _classification_metrics(all_targets, all_preds, num_classes)
    out.update({
        "acc": 100.0 * total_correct / seen,
        "loss": total_loss / seen,
    })
    return out

def comp_activity(net_g, dataset, args):
    net_g.eval()
    # testing
    data_loader = DataLoader(dataset, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        activity = torch.zeros(net_g(data, count_active_layers = True))
        break
    batch_count = 0
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        activity += torch.tensor(net_g(data, report_activity = True))
        # sum up batch loss
        batch_count += 1
    activity = activity/batch_count

    return activity
