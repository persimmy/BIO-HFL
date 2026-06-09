# -*- coding: utf-8 -*-
import math
from typing import Dict, Iterable, Optional

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

DEFAULTS = dict(
    E_AC=1e-13,
    E_MAC=3.2e-12,
    TX_POWER_W=0.1,
    NOISE_W=1.9952623149688784e-14,
    BW_MHZ=1.0,
)

_VGG9_CONV = [
    (3,   64, 3, 1, 1, False),  # conv1
    (64,  64, 3, 1, 1, True ),  # conv2 -> pool1
    (64, 128, 3, 1, 1, False),  # conv3
    (128,128, 3, 1, 1, True ),  # conv4 -> pool2
    (128,256, 3, 1, 1, False),  # conv5
    (256,256, 3, 1, 1, False),  # conv6
    (256,256, 3, 1, 1, True ),  # conv7 -> pool3
]
_VGG9_FC = [(4096, 1024), (1024, 10)]

def _sz_after(k: int, s: int, p: int, cur: int) -> int:
    return (cur - k + 2 * p) // s + 1


def vgg9_snn_acs_per_sample(T: int,
                            layer_rates: Optional[Iterable[float]],
                            num_classes: int = 10) -> int:
    """Estimate VGG9 SNN accumulate operations per sample."""
    default_rates = [0.22, 0.15, 0.45, 0.20, 0.35, 0.17, 0.15, 0.20, 0.06]
    rates = list(layer_rates) if layer_rates is not None else default_rates
    if len(rates) != 9:
        rates = default_rates

    cur, idx, acs = 32, 0, 0
    for ic, oc, k, s, p, pool in _VGG9_CONV:
        m = _sz_after(k, s, p, cur)
        ops = (m*m) * ic * (k*k) * oc
        acs += int(ops * rates[idx])
        cur = m // 2 if pool else m
        idx += 1
    # fc
    acs += int(4096 * 1024 * rates[idx]); idx += 1
    acs += int(1024 * int(num_classes) * rates[idx]); idx += 1
    return acs * max(1, int(T))

def _pick(k: str, args) -> float:
    """Read an energy or wireless-channel parameter from args or defaults."""
    if args is not None and hasattr(args, k.lower()):
        return float(getattr(args, k.lower()))
    alias = dict(E_AC='e_ac', E_MAC='e_mac', TX_POWER_W='tx_power_w', NOISE_W='noise_w', BW_MHZ='bw_mhz')
    if args is not None and alias.get(k) and hasattr(args, alias[k]):
        return float(getattr(args, alias[k]))
    return DEFAULTS[k]

def wireless_time_sec(bits: float, distance_m: float, args=None) -> float:
    bw_mhz  = _pick('BW_MHZ', args)
    ptx_w   = _pick('TX_POWER_W', args)
    noise_w = _pick('NOISE_W', args)
    distance_km = max(1.0, float(distance_m)) / 1000.0
    pathloss_db = 128.1 + 37.6 * math.log10(distance_km)
    h_lin = 10 ** (-pathloss_db / 10.0)
    snr = max(1e-12, (ptx_w * h_lin) / max(1e-20, noise_w))
    rate_bps = bw_mhz * 1e6 * math.log2(1.0 + snr)
    return float(bits) / max(1.0, rate_bps)

def comm_energy_from_bytes(num_bytes: int, distance_m: float, args=None) -> float:
    """Compute transmission energy for a payload sent over the wireless link."""
    t = wireless_time_sec(8.0 * max(0, int(num_bytes)), distance_m, args)
    return _pick('TX_POWER_W', args) * t


def compute_energy_snn(samples: int,
                       layer_rates: Optional[Iterable[float]],
                       args=None) -> float:
    T = getattr(args, 'timesteps', 10) if args is not None else 10
    ref_r = 6.0
    current_r = 6.0 * (T / 30.0)
    scale = ref_r/current_r
    num_classes = getattr(args, 'num_classes', 10) if args is not None else 10
    acs = vgg9_snn_acs_per_sample(T=T, layer_rates=layer_rates, num_classes=num_classes)
    return float(acs) * max(0, int(samples)) * _pick('E_AC', args)*scale

def state_dict_size_bytes(state: Dict[str, 'Tensor']) -> int:
    total = 0
    for v in state.values():
        if _HAS_TORCH and isinstance(v, torch.Tensor):
            total += int(v.numel()) * int(v.element_size())
        else:
            try:
                total += int(v.numel()) * 4
            except Exception:
                pass
    return total

def typical_client_distance_m(mean=80.0, std=20.0, lo=30.0, hi=150.0):
    import random
    d = random.gauss(mean, std)
    return float(min(max(d, lo), hi))

