from typing import Iterable, Sequence


def jain_index(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    denom = len(vals) * sum(v * v for v in vals)
    if denom <= 0.0:
        return 0.0
    total = sum(vals)
    return (total * total) / denom


def edge_average_jain(selection_counts: Sequence[float], edge_servers) -> float:
    scores = []
    for edge in edge_servers:
        counts = [selection_counts[int(gid)] for gid in edge.clients_idx]
        scores.append(jain_index(counts))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
