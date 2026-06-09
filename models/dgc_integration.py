# models/dgc_integration.py
import math
from typing import Dict, Tuple, Any, Iterable
import torch



def _tensor_nbytes(x: torch.Tensor) -> int:
    return x.numel() * (2 if x.dtype == torch.float16 else
                        4 if x.dtype in (torch.float32, torch.int32) else
                        x.element_size())


class _FallbackTopKWithEF:
    def __init__(self,
                 keep_ratio_base: float,
                 warmup_epochs: int,
                 fp16_values: bool,
                 res_decay: float = 1.0,
                 res_clip_norm: float | None = None,
                 enable_stats: bool = False,
                 stats_reset_every: int = 1,
                 critical_tensor_enable: bool = True):
        if keep_ratio_base <= 1.0:
            self.keep_ratio_target = float(keep_ratio_base)
        else:
            self.keep_ratio_target = 1.0 / float(keep_ratio_base)
        self.keep_ratio_target = max(min(self.keep_ratio_target, 1.0), 0.0)

        self.warmup_epochs = int(max(0, warmup_epochs))
        self.fp16_values = bool(fp16_values)

        self.res_decay = float(res_decay)
        self.res_clip_norm = (float(res_clip_norm)
                              if res_clip_norm is not None else None)

        self.keep_ratio_cur = 1.0
        self._residuals: Dict[str, torch.Tensor] = {}
        self._shapes: Dict[str, torch.Size] = {}
        self.enable_stats = bool(enable_stats)
        self.stats_reset_every = int(max(1, stats_reset_every))
        self._stats_round = {"send_l2": [], "res_l2": []}
        self._round_idx = 0

        self.crit_period = 1 if critical_tensor_enable else 0
        self.crit_name_patterns = ("fc2.weight", "bntt_fc.", "bntt7.")
        self.crit_max_elems = 10240

        self.crit_verbose = False

    def _is_critical_tensor(self, name: str, numel: int) -> bool:
        if self.crit_period is None or self.crit_period <= 0:
            return False
        if numel > int(self.crit_max_elems):
            return False

        nm = name.lower()
        shape = self._shapes.get(name, None)

        if "fc2.weight" in nm:
            return True

        if ("bntt_fc." in nm) or ("bntt7." in nm):
            if shape is not None and len(shape) == 1:
                return True

        return False

    # in class _FallbackTopKWithEF:
    def set_epoch(self, epoch: int):
        """Update the active keep ratio according to the DGC warmup schedule."""
        if self.warmup_epochs <= 0:
            self.keep_ratio_cur = float(self.keep_ratio_target)
            return

        t = max(0.0, min(1.0, float(epoch) / float(self.warmup_epochs)))
        target = max(self.keep_ratio_target, 1e-6)
        self.keep_ratio_cur = float(target ** t)

    def initialize_from_state(self, state):
        self._residuals = {}
        self._shapes = {}
        for name, tensor in state.items():
            if not tensor.is_floating_point():
                continue
            self._residuals[name] = torch.zeros_like(tensor, dtype=torch.float32, device='cpu')
            self._shapes[name] = tensor.shape

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != 'cpu':
            x = x.detach().cpu()
        return x.to(dtype=torch.float32).contiguous().view(-1)

    def _unflatten(self, flat: torch.Tensor, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        out = flat.view(shape).to(dtype=dtype)
        return out

    def compress(self, deltas: Dict[str, torch.Tensor]) -> Dict[
        str, Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]]:
        """Compress trainable parameter deltas into sparse or dense payloads.

        Values are stored in fp16 or fp32, indices are int32, and the context
        records tensor shape, dtype, and whether the payload is dense.
        """
        payload = {}

        for name, delta in deltas.items():
            shape = delta.shape
            base_dtype = torch.float32
            device_cpu = "cpu"

            if (name not in self._residuals):
                v = self._flatten(delta)  # f32, 1-D, CPU
                values = v.to(dtype=torch.float16) if self.fp16_values else v
                idx = torch.arange(values.numel(), dtype=torch.int32)
                ctx = dict(shape=shape, dtype=base_dtype, dense=True)
                payload[name] = ((values, idx), ctx)
                continue

            res = self._residuals[name].view(-1)  # f32, cpu
            if self.res_decay < 1.0:
                res.mul_(self.res_decay)
            g = self._flatten(delta) + res  # f32, 1-D, CPU
            N = g.numel()

            do_crit_dense = (
                    self.crit_period is not None and self.crit_period > 0 and
                    (self._round_idx > 0) and (self._round_idx % int(self.crit_period) == 0) and
                    self._is_critical_tensor(name, N)
            )
            if do_crit_dense:
                values_f32 = g
                values = values_f32.to(dtype=torch.float16) if self.fp16_values else values_f32
                idx = torch.arange(N, dtype=torch.int32)
                self._residuals[name].zero_()
                dense = True

                if self.enable_stats:
                    self._stats_round["send_l2"].append(float(values_f32.norm(p=2).item()))
                    self._stats_round["res_l2"].append(0.0)

                if self.crit_verbose:
                    print(f"[DGC][CritDense][round={self._round_idx}] name={name} N={N} fp16={self.fp16_values}")

                ctx = dict(shape=shape, dtype=base_dtype, dense=dense)
                payload[name] = ((values, idx), ctx)
                continue

            keep_ratio = float(self.keep_ratio_cur)
            if keep_ratio >= 1.0 or N == 0:
                k = N
            else:
                k = max(1, int(math.ceil(N * keep_ratio)))

            if k >= N:
                values_f32 = g
                values = values_f32.to(dtype=torch.float16) if self.fp16_values else values_f32
                idx = torch.arange(N, dtype=torch.int32)
                self._residuals[name].zero_()
                dense = True
                if self.enable_stats:
                    send_l2 = float(values_f32.norm(p=2).item())
                    res_l2 = 0.0
                    self._stats_round["send_l2"].append(send_l2)
                    self._stats_round["res_l2"].append(res_l2)

            else:
                topk = torch.topk(g.abs(), k, largest=True, sorted=False)
                indices = topk.indices.to(dtype=torch.int64)
                values_f32 = g.index_select(0, indices)
                values = values_f32.to(dtype=torch.float16) if self.fp16_values else values_f32
                idx = indices.to(dtype=torch.int32)
                scatter = torch.zeros_like(g)
                scatter.index_copy_(0, indices, values_f32)
                new_res = (g - scatter)

                if self.res_clip_norm is not None:
                    n = torch.norm(new_res, p=2).item() + 1e-12
                    if n > self.res_clip_norm:
                        new_res.mul_(self.res_clip_norm / n)

                self._residuals[name] = new_res.view_as(self._residuals[name])
                dense = False
                if self.enable_stats:
                    send_l2 = float(values_f32.norm(p=2).item())
                    res_l2 = float(new_res.norm(p=2).item())
                    self._stats_round["send_l2"].append(send_l2)
                    self._stats_round["res_l2"].append(res_l2)
            ctx = dict(shape=shape, dtype=base_dtype, dense=dense)
            payload[name] = ((values, idx), ctx)

        return payload

    def stats_report_and_reset(self):
        """Print and reset optional DGC residual statistics."""
        if not self.enable_stats:
            return

        try:
            import numpy as np
        except Exception:
            def _brief(arr):
                if not arr:
                    return "count=0"
                s = sum(arr);
                n = len(arr)
                mean = s / n
                mx = max(arr)
                return f"count={n} mean={mean:.4g} max={mx:.4g}"

            msg = (f"[DGC][Stats][round={self._round_idx}] "
                   f"send_l2: {_brief(self._stats_round['send_l2'])} | "
                   f"res_l2:  {_brief(self._stats_round['res_l2'])}")
            print(msg)
        else:
            def _summary(arr):
                if len(arr) == 0:
                    return "count=0"
                a = np.asarray(arr, dtype=np.float64)
                p50 = np.percentile(a, 50)
                p90 = np.percentile(a, 90)
                p95 = np.percentile(a, 95)
                p99 = np.percentile(a, 99)
                return (f"count={len(a)} mean={a.mean():.4g} "
                        f"p50={p50:.4g} p90={p90:.4g} p95={p95:.4g} p99={p99:.4g} max={a.max():.4g}")

            msg = (f"[DGC][Stats][round={self._round_idx}] "
                   f"send_l2: {_summary(self._stats_round['send_l2'])} | "
                   f"res_l2:  {_summary(self._stats_round['res_l2'])}")
            print(msg)

        self._stats_round['send_l2'].clear()
        self._stats_round['res_l2'].clear()

    def decompress_and_sum(self,
                           payloads_list: Iterable[Dict[str, Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]]],
                           weights: Iterable[float],
                           template_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reconstruct sparse payloads and return their weighted dense sum."""
        agg: Dict[str, torch.Tensor] = {k: torch.zeros_like(v, dtype=v.dtype, device='cpu') for k, v in template_state.items()}
        weights = list(weights)
        for payload, w in zip(payloads_list, weights):
            for name, ((values, idx), ctx) in payload.items():
                shape = ctx['shape']
                dense = ctx['dense']
                if dense:
                    flat = values.to(dtype=torch.float32)
                else:
                    flat = torch.zeros(int(torch.prod(torch.tensor(shape))), dtype=torch.float32)
                    flat.index_copy_(0, idx.to(dtype=torch.int64), values.to(dtype=torch.float32))
                agg[name] += w * flat.view(shape).to(dtype=template_state[name].dtype)
        return agg

    def payload_nbytes(self, payload):
        total = 0
        for ((values, idx), ctx) in payload.values():
            if ctx.get('dense', False):
                total += _tensor_nbytes(values)
            else:
                total += _tensor_nbytes(values) + _tensor_nbytes(idx)
        return total
# ==========================================================================


class DGCManager:
    def __init__(self,
                 compress_ratio: float = 600.0,
                 warmup_epochs: int = 0,
                 fp16_values: bool = False,
                 res_decay: float = 1.0,
                 res_clip_norm: float | None = None,
                 enable_stats: bool = False,
                 stats_reset_every: int = 1,
                 res_clip_warm: float | None = None,
                 critical_tensor_enable: bool = True):
        self.compress_ratio = float(compress_ratio)
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.fp16 = bool(fp16_values)
        self.res_decay = float(res_decay)
        self.res_clip_norm = (float(res_clip_norm) if res_clip_norm is not None else None)
        self.res_clip_warm = (float(res_clip_warm) if res_clip_warm is not None else None)

        self.compressor = _FallbackTopKWithEF(
            keep_ratio_base=self.compress_ratio,
            warmup_epochs=self.warmup_epochs,
            fp16_values=self.fp16,
            res_decay=self.res_decay,
            res_clip_norm=self.res_clip_norm,
            enable_stats=enable_stats,
            stats_reset_every=stats_reset_every,
            critical_tensor_enable=critical_tensor_enable,
        )
        self._is_fallback = True


    def end_round(self):
        """Finalize a round and flush optional DGC statistics."""
        comp = getattr(self, "compressor", None)
        if comp is not None and getattr(comp, "enable_stats", False):
            comp.stats_report_and_reset()

    def initialize_from_model(self, model: torch.nn.Module):
        state = {n: p.data.detach().cpu() for (n, p) in model.named_parameters() if p.requires_grad}
        self.compressor.initialize_from_state(state)
        import os, io, datetime

        if not hasattr(self, "_dumped_names") or not self._dumped_names:
            result_dir = getattr(self, "result_dir", None) or "result"

            if result_dir is None:
                result_dir = "result"
            os.makedirs(result_dir, exist_ok=True)
            dump_path = os.path.join(result_dir, "param_names.txt")

            buf = io.StringIO()
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            buf.write(f"[Dump @ {ts}]\n\n")

            buf.write("[Named Parameters]\n")
            for n, p in model.named_parameters():
                shape = tuple(p.shape)
                buf.write(f"  {n}  shape={shape}  trainable={p.requires_grad}\n")
            buf.write("\n[Named Buffers]\n")
            for n, b in model.named_buffers():
                shape = tuple(b.shape)
                dtype = str(b.dtype)
                buf.write(f"  {n}  shape={shape}  dtype={dtype}\n")

            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(buf.getvalue())

            print(f"[DGC][Debug] param/buffer names dumped to: {dump_path}")
            self._dumped_names = True

    # in class DGCManager:
    def set_epoch(self, epoch: int):
        """Advance DGC warmup, residual decay, and residual clipping schedules."""
        # in DGCManager.set_epoch(self, epoch: int)
        if self._is_fallback:
            self.compressor.set_epoch(epoch)
            if hasattr(self.compressor, "_round_idx"):
                self.compressor._round_idx = int(epoch) + 1

            decay_target = float(getattr(self, "res_decay", 1.0))
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                t = float(epoch) / float(self.warmup_epochs)  # [0,1)
                decay_tau = 1.0 - (1.0 - decay_target) * t
            else:
                decay_tau = decay_target
            if hasattr(self.compressor, "res_decay"):
                self.compressor.res_decay = float(decay_tau)

            target = self.res_clip_norm
            warm = self.res_clip_warm if self.res_clip_warm is not None else 0.18

            if target is None:
                self.compressor.res_clip_norm = None
            else:
                if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                    t = float(epoch) / float(self.warmup_epochs)  # [0,1)
                    tau = warm * (1.0 - t) + float(target) * t
                else:
                    tau = float(target)
                self.compressor.res_clip_norm = float(tau)

            # if hasattr(self.compressor, "begin_round"):
            #     self.compressor.begin_round()
            try:
                cur_keep = getattr(self.compressor, "keep_ratio_cur", None)
                cur_decay = getattr(self.compressor, "res_decay", None)
                cur_clip = getattr(self.compressor, "res_clip_norm", None)
                cur_round = getattr(self.compressor, "_round_idx", epoch + 1)
                print(f"[DGC][Warmup] round={int(cur_round)} "
                      f"keep_ratio={cur_keep:.6f} "
                      f"res_decay={('None' if cur_decay is None else f'{cur_decay:.6f}')} "
                      f"res_clip_norm={('None' if cur_clip is None else f'{cur_clip:.6f}')}")
            except Exception:
                pass


    def compress_deltas(self, deltas: Dict[str, torch.Tensor]):
        return self.compressor.compress(deltas)

    def decompress_and_weighted_sum(self,
                                    payloads_list: Iterable[Dict[str, Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]]],
                                    weights: Iterable[float],
                                    template_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.compressor.decompress_and_sum(payloads_list, weights, template_state)

    def payload_nbytes(self, payload):
        total = 0
        for ((values, idx), ctx) in payload.values():
            if ctx.get('dense', False):
                total += _tensor_nbytes(values)
            else:
                total += _tensor_nbytes(values) + _tensor_nbytes(idx)
        return total

