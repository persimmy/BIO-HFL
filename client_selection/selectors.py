import math
from typing import Dict, Iterable, List, Optional


class ClientSelectionManager:
    """Thin adapter around the project's client-selection strategies.

    Existing MAB, Greedy, and Gossip behavior stays in EdgeServer. Oort-loss is
    implemented here so the original model/edge classes remain mostly intact.
    """

    SUPPORTED = {"mab", "dmab", "greedy", "gossip", "oort_loss"}

    def __init__(self, args):
        method = str(getattr(args, "client_selection", "gossip")).lower()
        if method == "oort-loss":
            method = "oort_loss"
        if method not in self.SUPPORTED:
            raise ValueError(
                "Unsupported client_selection '{}'. Choose from {}.".format(
                    method, sorted(self.SUPPORTED)
                )
            )
        self.method = "mab" if method == "dmab" else method
        self.args = args
        self.latest_loss: Dict[int, float] = {}

    def select(self, edge, global_round: int, selection_counts: Iterable[int]) -> List[int]:
        if self.method == "mab":
            return edge.MAB_selection2(current_epoch=global_round + 1)
        if self.method == "greedy":
            return edge.greedy_selection()
        if self.method == "gossip":
            return edge.gossip_selection()
        if self.method == "oort_loss":
            return self._select_oort_loss(edge, selection_counts)
        raise AssertionError("unreachable")

    def update_losses(self, per_client_loss: Dict[int, float]) -> None:
        for gid, loss in per_client_loss.items():
            if loss is None:
                continue
            loss_f = float(loss)
            if math.isfinite(loss_f):
                self.latest_loss[int(gid)] = loss_f

    def scores_for_edge(self, edge) -> Dict[int, Optional[float]]:
        return {int(gid): self.latest_loss.get(int(gid)) for gid in edge.clients_idx}

    def _select_oort_loss(self, edge, selection_counts: Iterable[int]) -> List[int]:
        k = min(int(getattr(self.args, "mab_k", 1)), len(edge.clients_idx))
        counts = list(selection_counts)
        candidates = [int(gid) for gid in edge.clients_idx]

        unknown = [gid for gid in candidates if gid not in self.latest_loss]
        known = [gid for gid in candidates if gid in self.latest_loss]

        selected: List[int] = []
        # Bootstrap phase: try clients without an observed loss first, so Oort
        # has a real score for every client instead of silently ignoring them.
        unknown.sort(key=lambda gid: (counts[gid] if gid < len(counts) else 0, gid))
        selected.extend(unknown[:k])

        if len(selected) < k:
            remaining = [gid for gid in known if gid not in selected]
            remaining.sort(
                key=lambda gid: (
                    -float(self.latest_loss[gid]),
                    counts[gid] if gid < len(counts) else 0,
                    gid,
                )
            )
            selected.extend(remaining[: k - len(selected)])

        return selected
