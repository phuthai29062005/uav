"""Crossed 5x20 bootstrap and P1 support logic for rank-reversal diagnostic (E4P/E4PI)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def crossed_bootstrap_g_rev(units: List[Dict],
                            policy_seeds: List[int],
                            env_seeds: List[int],
                            scenarios: List[str],
                            n_resamples: int = 50000,
                            seed: int = 126001) -> Dict:
    rng = np.random.RandomState(seed)
    n_pol = len(policy_seeds)
    n_env = len(env_seeds)
    pol_indices = rng.randint(0, n_pol, size=(n_resamples, n_pol))
    env_indices = rng.randint(0, n_env, size=(n_resamples, n_env))

    lookup = {}
    for u in units:
        key = (u["policy_seed"], u["env_seed"], u["scenario"])
        lookup[key] = u

    g_rev_boots = []
    g_bac_boots = []
    for b in range(n_resamples):
        pols = [policy_seeds[i] for i in pol_indices[b]]
        envs = [env_seeds[i] for i in env_indices[b]]
        scn_rr = {s: [] for s in scenarios}
        scn_bac = {s: [] for s in scenarios}
        for p in pols:
            for e in envs:
                for s in scenarios:
                    u = lookup.get((p, e, s))
                    if u is not None and u.get("unit_rr") is not None:
                        scn_rr[s].append(u["unit_rr"])
                    if u is not None and u.get("unit_bac") is not None:
                        scn_bac[s].append(u["unit_bac"])
        rr_means = {}
        bac_means = {}
        for s in scenarios:
            rr_means[s] = float(np.mean(scn_rr[s])) if scn_rr[s] else None
            bac_means[s] = float(np.mean(scn_bac[s])) if scn_bac[s] else None
        if all(rr_means[s] is not None for s in scenarios):
            g = rr_means["STATIONARY"] - 0.5 * (rr_means["PERSISTENT"] + rr_means["MIXED"])
            g_rev_boots.append(g)
        if all(bac_means[s] is not None for s in scenarios):
            g = bac_means["STATIONARY"] - 0.5 * (bac_means["PERSISTENT"] + bac_means["MIXED"])
            g_bac_boots.append(g)

    g_rev_arr = np.array(g_rev_boots) if g_rev_boots else np.array([])
    g_bac_arr = np.array(g_bac_boots) if g_bac_boots else np.array([])
    result = {
        "n_resamples": n_resamples,
        "seed": seed,
        "n_valid_g_rev": len(g_rev_boots),
        "n_valid_g_bac": len(g_bac_boots),
    }
    if len(g_rev_boots) > 0:
        result["G_REV_mean"] = float(np.mean(g_rev_arr))
        result["G_REV_CI_lower"] = float(np.percentile(g_rev_arr, 2.5))
        result["G_REV_CI_upper"] = float(np.percentile(g_rev_arr, 97.5))
    if len(g_bac_boots) > 0:
        result["G_BAC_mean"] = float(np.mean(g_bac_arr))
        result["G_BAC_CI_lower"] = float(np.percentile(g_bac_arr, 2.5))
        result["G_BAC_CI_upper"] = float(np.percentile(g_bac_arr, 97.5))
    return result


def compute_policy_gaps(units: List[Dict],
                        policy_seeds: List[int],
                        scenarios: List[str]) -> Dict:
    result = {}
    for p in policy_seeds:
        p_units = [u for u in units if u["policy_seed"] == p]
        scn_rr = {}
        scn_bac = {}
        for s in scenarios:
            s_units = [u for u in p_units if u["scenario"] == s]
            rr_vals = [u["unit_rr"] for u in s_units if u.get("unit_rr") is not None]
            bac_vals = [u["unit_bac"] for u in s_units if u.get("unit_bac") is not None]
            scn_rr[s] = float(np.mean(rr_vals)) if rr_vals else None
            scn_bac[s] = float(np.mean(bac_vals)) if bac_vals else None
        g_rev = None
        g_bac = None
        if all(scn_rr[s] is not None for s in scenarios):
            g_rev = scn_rr["STATIONARY"] - 0.5 * (scn_rr["PERSISTENT"] + scn_rr["MIXED"])
        if all(scn_bac[s] is not None for s in scenarios):
            g_bac = scn_bac["STATIONARY"] - 0.5 * (scn_bac["PERSISTENT"] + scn_bac["MIXED"])
        result[p] = {"G_REV": g_rev, "G_BAC": g_bac}
    return result


def compute_source_arm_g_rev(units: List[Dict],
                             scenarios: List[str]) -> Dict:
    result = {}
    for arm in ["C1", "C2"]:
        scn_rr = {}
        for s in scenarios:
            arm_units = [u for u in units
                         if u.get("source_arm") == arm and u["scenario"] == s]
            rr_vals = [u.get("rr") for u in arm_units if u.get("rr") is not None]
            scn_rr[s] = float(np.mean(rr_vals)) if rr_vals else None
        g = None
        if all(scn_rr[s] is not None for s in scenarios):
            g = scn_rr["STATIONARY"] - 0.5 * (scn_rr["PERSISTENT"] + scn_rr["MIXED"])
        result[arm] = {"G_REV": g, "scenario_rr": scn_rr}
    return result


def classify_p1(g_rev: Optional[float],
                bootstrap_ci_lower: Optional[float],
                policy_gaps: Dict,
                g_bac: Optional[float],
                policy_bac_gaps: Dict,
                source_arm_g_rev: Dict,
                n_valid_surfaces: int = 100) -> Dict:
    if n_valid_surfaces < 10:
        return {"verdict": "NOT_AUDITABLE", "reason": "insufficient_valid_surfaces",
                "criteria": {}}
    criteria = {}
    criteria["C1_G_REV_positive"] = g_rev is not None and g_rev > 0
    criteria["C2_CI_lower_positive"] = (bootstrap_ci_lower is not None
                                        and bootstrap_ci_lower > 0)
    pol_g_revs = [v["G_REV"] for v in policy_gaps.values() if v["G_REV"] is not None]
    n_pos_pol = sum(1 for g in pol_g_revs if g > 0)
    criteria["C3_policy_4_of_5"] = n_pos_pol >= 4
    criteria["C4_G_BAC_positive"] = g_bac is not None and g_bac > 0
    pol_g_bacs = [v["G_BAC"] for v in policy_gaps.values() if v["G_BAC"] is not None]
    n_pos_bac = sum(1 for g in pol_g_bacs if g > 0)
    criteria["C5_policy_BAC_4_of_5"] = n_pos_bac >= 4
    c1_g = source_arm_g_rev.get("C1", {}).get("G_REV")
    c2_g = source_arm_g_rev.get("C2", {}).get("G_REV")
    criteria["C6_both_sources_positive"] = (c1_g is not None and c1_g > 0
                                            and c2_g is not None and c2_g > 0)
    all_pass = all(criteria.values())
    any_positive = (criteria.get("C1_G_REV_positive", False)
                    or any(g > 0 for g in pol_g_revs))
    if all_pass:
        verdict = "SUPPORTED"
    elif any_positive:
        verdict = "PLAUSIBLE"
    else:
        if g_rev is not None and g_rev < 0:
            verdict = "NOT_SUPPORTED"
        else:
            verdict = "NOT_SUPPORTED"
    return {"verdict": verdict, "criteria": criteria,
            "n_positive_policy_G_REV": n_pos_pol,
            "n_positive_policy_G_BAC": n_pos_bac}
