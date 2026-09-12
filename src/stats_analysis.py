"""
5.2 — Statistical analysis primitives for the frozen 14x30 benchmark.

Pure functions only (no I/O), so they can be unit-tested independently
of the frozen dataset.
"""
import numpy as np
from scipy.stats import rankdata, wilcoxon


def holm_correction(pvals, alpha=0.05):
    """
    Holm step-down correction (mathematically equivalent to
    statsmodels.stats.multitest.multipletests(method="holm")).

    return: (p_holm: list[float] in ORIGINAL order, reject: list[bool])
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals, kind="stable")          # ascending
    sorted_p = pvals[order]

    adj = (m - np.arange(m)) * sorted_p                # (m-i+1)*p_(i), 0-indexed i
    adj = np.maximum.accumulate(adj)                   # enforce monotonicity
    adj = np.clip(adj, 0.0, 1.0)

    p_holm = np.empty(m)
    p_holm[order] = adj
    reject = p_holm < alpha
    return p_holm.tolist(), reject.tolist()


def rank_biserial(bl, sa):
    """
    Paired rank-biserial correlation for d = BL - SA.
    d > 0 -> SA has lower MIGD -> r_rb > 0 favors SA.
    Zero differences excluded from ranking (matches Wilcoxon zero_method="wilcox").
    Ties in |d| get average ranks.

    return: dict(W_plus, W_minus, r_rb, n_nonzero)
    """
    bl = np.asarray(bl, dtype=float)
    sa = np.asarray(sa, dtype=float)
    d = bl - sa
    nz = d[d != 0]
    if len(nz) == 0:
        return dict(W_plus=0.0, W_minus=0.0, r_rb=0.0, n_nonzero=0)

    ranks = rankdata(np.abs(nz), method="average")
    W_plus = float(ranks[nz > 0].sum())
    W_minus = float(ranks[nz < 0].sum())
    denom = W_plus + W_minus
    r_rb = 0.0 if denom == 0 else (W_plus - W_minus) / denom
    return dict(W_plus=W_plus, W_minus=W_minus, r_rb=r_rb, n_nonzero=len(nz))


def wilcoxon_paired(bl, sa):
    """
    Paired two-sided Wilcoxon signed-rank test on d = BL - SA,
    zero_method="wilcox" (exact zeros excluded from ranking).

    return: dict(W, p_raw, n_nonzero)
    """
    bl = np.asarray(bl, dtype=float)
    sa = np.asarray(sa, dtype=float)
    d = bl - sa
    n_nonzero = int(np.sum(d != 0))
    if n_nonzero == 0:
        return dict(W=0.0, p_raw=1.0, n_nonzero=0)
    res = wilcoxon(bl, sa, alternative="two-sided", zero_method="wilcox")
    return dict(W=float(res.statistic), p_raw=float(res.pvalue), n_nonzero=n_nonzero)


def effect_magnitude(r_rb):
    """Descriptive (not pass/fail) magnitude label for |r_rb|."""
    a = abs(r_rb)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "small"
    if a < 0.5:
        return "moderate"
    return "large"


def classify_primary(p_holm, median_d, alpha=0.05):
    """
    Pre-fixed classification rule (Phan 7):
    p_holm<alpha & median_d>0 -> SA+
    p_holm<alpha & median_d<0 -> SA-
    p_holm<alpha & median_d==0 -> flagged mixed-zero-median
    else -> "="
    """
    if p_holm < alpha:
        if median_d > 0:
            return "SA+"
        if median_d < 0:
            return "SA-"
        return "significant/mixed-zero-median"
    return "="
