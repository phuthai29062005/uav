"""
5.3A — pure diagnostic helpers for frozen-policy action/state mechanism
analysis. No I/O, no production side effects.
"""
import numpy as np

GATE_LABELS = {0: "NO_MEMORY", 1: "MEMORY"}
SEG_LABELS = {0: "KEEP", 1: "LOCAL", 2: "PREDICT", 3: "DIVERSIFY"}


def decode_gate(g):
    return GATE_LABELS[int(g)]


def decode_seg(a):
    return SEG_LABELS[int(a)]


def normalized_entropy(labels, n_categories):
    """
    Shannon entropy of the empirical distribution of `labels`, normalized
    by log(n_categories) -> [0,1]. 0 = always same category, 1 = uniform.
    Empty input -> nan (undefined).
    """
    labels = list(labels)
    if len(labels) == 0:
        return float("nan")
    counts = np.zeros(n_categories)
    for l in labels:
        counts[int(l)] += 1
    p = counts / counts.sum()
    p_nz = p[p > 0]
    H = -np.sum(p_nz * np.log(p_nz))
    if n_categories <= 1:
        return 0.0
    return float(H / np.log(n_categories))


def tercile_labels(values):
    """
    Deterministic within-array tercile split into LOW/MID/HIGH by rank
    (equal-count groups via argsort, ties broken by stable original order).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    cut1, cut2 = n // 3, (2 * n) // 3
    labels = np.empty(n, dtype=object)
    labels[ranks < cut1] = "LOW"
    labels[(ranks >= cut1) & (ranks < cut2)] = "MID"
    labels[ranks >= cut2] = "HIGH"
    return labels


def higher_c_segment(c1, c2):
    """1 if c1>=c2 (segment 1 changed more, tie->segment1), else 2."""
    return 1 if c1 >= c2 else 2


def intervention_rate(seg_actions):
    """Fraction of NO_MEMORY segment decisions that are != KEEP(0)."""
    seg_actions = list(seg_actions)
    if not seg_actions:
        return float("nan")
    return float(np.mean([a != 0 for a in seg_actions]))


def window_label(change_index, early=(1, 10), mid=(46, 55), late=(91, 100)):
    """change_index is 1-based (matches timeline's change_index)."""
    if early[0] <= change_index <= early[1]:
        return "EARLY"
    if mid[0] <= change_index <= mid[1]:
        return "MID"
    if late[0] <= change_index <= late[1]:
        return "LATE"
    return "OTHER"
