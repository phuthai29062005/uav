"""
E2a -- Controlled Regional Operator Oracle.

Trả lời: (1) các region có thực sự cần response operator khác nhau
không (operator heterogeneity), và (2) regional operator selection có
tốt hơn một global operator áp cho toàn population không (actionable
regional gain, ROG).

FREEZE E1 (bắt buộc, xem task): KHÔNG import/sửa partition.py,
severity.py, rhi.py, quality_loss.py theo hướng thay đổi hành vi của
chúng -- severity CHỈ được tính để GHI vào output (occupancy/severity
diagnostic), KHÔNG được dùng để chọn action ở bất kỳ đâu trong file
này.

KHÔNG dùng quality_loss làm Q (Section 2): Q = -IGD(F, PF*), PF* lấy
từ prdrl.problems.controlled_operator.get_true_pf() (KHÔNG đổi theo t
-- xem docstring ở đó).

KHÔNG dùng oracle.py cũ (prdrl/oracle/oracle.py): module đó phục vụ
pipeline E2 (CEC-based). Chỉ tái dùng hàm thuần cơ học
nsga2_survival từ oracle.py.

Current PDF protocol:
    mu^{t-2}, mu^{t-1}  ->  response tại environment t

Missing history:
    PREDICT falls back to LOCAL using sigma_local
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, "src")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

from prdrl.core.normalize import normalize
from prdrl.core.reference import reference
from prdrl.core.partition import partition
from prdrl.core.seeding import deterministic_rng
from prdrl.detectors.severity import severity          # chi de GHI, khong dung de chon action
from prdrl.metrics.igd import igd

from prdrl.operators.keep import keep
from prdrl.operators.local import local
from prdrl.operators.predict import predict
from prdrl.operators.diversify import diversify
from prdrl.operators.budget import equal_budget

from prdrl.oracle.oracle import nsga2_survival

from prdrl.history.region_history import RegionHistory
from prdrl.features.forecast_state import get_predict_motion_from_history

from prdrl.problems.controlled_operator import (
    evaluate as op_evaluate, get_bounds, get_true_pf,
    make_region_dynamics_map, SCENARIOS,
)


# ============================================================
# LOCKED CONFIG (Section 5/6 -- KHONG tune sau khi xem ket qua)
# ============================================================

ACTIONS = ("KEEP", "LOCAL", "PREDICT", "DIVERSIFY")

SIGMA_LOCAL = 0.02
SIGMA_PREDICT = 0.03
SIGMA_DIVERSIFY = 0.15
LAMBDA_PREDICT = 1.0
BUDGET = 10

N_OBJ = 2
D = 10
POP_SIZE = 100
N_PARTITIONS = 9         # M=2 -> R=10 (Das-Dennis, KHONG phai so region)
WARMUP_GENERATIONS = 30
GENERATIONS_PER_CHANGE = 10
N_T = 10


def compute_time(generation, n_t=N_T, tau_t=10):
    return (1.0 / n_t) * np.floor(generation / tau_t)


class OperatorProblem(Problem):
    def __init__(self, t, region_dynamics, xl, xu):
        super().__init__(n_var=len(xl), n_obj=N_OBJ, xl=xl, xu=xu)
        self.t = t
        self.region_dynamics = region_dynamics

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = op_evaluate(X, self.t, self.region_dynamics)


def evolve_nsga2(X_start, t, region_dynamics, n_gen, seed, xl, xu):
    pop_size = len(X_start)
    problem = OperatorProblem(t, region_dynamics, xl, xu)
    algorithm = NSGA2(pop_size=pop_size, sampling=X_start)
    result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed,
                      verbose=False)
    return np.asarray(result.pop.get("X"), dtype=float)


# ============================================================
# Deterministic RNG: base_seed, event_id (= change_idx), region_id,
# action -- cùng triple (event_id, region_id, action) sinh cùng
# random stream dù được gọi trong isolated/global/regional trial.
# ============================================================

_ACTION_CODE = {"KEEP": 0, "LOCAL": 1, "PREDICT": 2, "DIVERSIFY": 3}


def make_rng(base_seed, event_id, region_id, action):
    ss = np.random.SeedSequence([int(base_seed), int(event_id),
                                 int(region_id), _ACTION_CODE[action]])
    return np.random.default_rng(ss)


# ============================================================
# Offspring generation for ONE region.
#
# Current PDF protocol:
#   mu^{t-2}, mu^{t-1} -> delta_hat -> PREDICT at environment t
#
# Missing history (delta_hat=None):
#   PREDICT falls back to LOCAL using sigma_local
# ============================================================

def generate_children(action, region_id, X_region, history, t,
                      budget, xl, xu, base_seed, event_id):
    D = len(xl)
    if len(X_region) == 0:
        return np.empty((0, D))

    if action == "KEEP":
        return keep(X_region, budget, xl, xu, 0.0, None)

    n_offspring = equal_budget(action, len(X_region), budget)
    rng = make_rng(base_seed, event_id, region_id, action)

    if action == "LOCAL":
        return local(X_region, n_offspring, xl, xu, SIGMA_LOCAL, rng)

    if action == "DIVERSIFY":
        return diversify(X_region, n_offspring, xl, xu, SIGMA_DIVERSIFY, rng)

    if action == "PREDICT":
        motion = get_predict_motion_from_history(
            history=history,
            region_id=region_id,
            t=t,
            lambda_pred=LAMBDA_PREDICT,
        )
        delta_hat = None if motion is None else motion["delta_hat"]
        return predict(
            X_prev=X_region,
            delta_hat=delta_hat,
            n_offspring=n_offspring,
            xl=xl,
            xu=xu,
            sigma_pred=SIGMA_PREDICT,
            rng=rng,
            sigma_local=SIGMA_LOCAL,
        )

    raise ValueError(f"Unknown action: {action}")


# ============================================================
# Respond: apply an action-map to the population, evaluate,
# survive back to N (Section 9, 11, 13).
# ============================================================

def respond(X_base, F_plus, region_ids, R, actions, history, t,
            evaluate_fn, xl, xu, budget, base_seed, event_id):
    X_base = np.asarray(X_base, dtype=float)
    F_plus = np.asarray(F_plus, dtype=float)
    region_ids = np.asarray(region_ids)
    N = len(X_base)

    children_all = []
    for r in range(R):
        mask = region_ids == r
        X_region = X_base[mask]
        if len(X_region) == 0:
            continue
        action = actions.get(r, "KEEP")
        children = generate_children(action, r, X_region, history, t,
                                     budget, xl, xu, base_seed, event_id)
        if len(children) > 0:
            children_all.append(children)

    if len(children_all) == 0:
        return {"available": True, "X": X_base.copy(), "F": F_plus.copy(),
               "extra_fe": 0}

    X_children = np.vstack(children_all)
    F_children = evaluate_fn(X_children)
    extra_fe = len(X_children)

    X_pool = np.vstack([X_base, X_children])
    F_pool = np.vstack([F_plus, F_children])
    X_new, F_new = nsga2_survival(X_pool, F_pool, n_survive=N, xl=xl, xu=xu)

    return {"available": True, "X": X_new, "F": F_new, "extra_fe": extra_fe}


def operator_heterogeneity_index(preferred_actions, eps=1e-12):
    if len(preferred_actions) < 2:
        return 0.0
    counts = np.array([sum(a == action for a in preferred_actions)
                       for action in ACTIONS], dtype=float)
    p = counts / counts.sum()
    nonzero = p > 0
    entropy = -np.sum(p[nonzero] * np.log(p[nonzero] + eps))
    return float(entropy / np.log(len(ACTIONS)))


# ============================================================
# ONE scored event: isolated trials -> OHI -> best global ->
# regional oracle -> ROG (Section 11-15).
# ============================================================

def run_oracle_event(X_base, F_plus, region_ids, R, history, t,
                     evaluate_fn, pf_star, xl, xu, budget, base_seed,
                     event_id):

    def quality_fn(F):
        return -igd(F, pf_star)

    keep_result = respond(X_base, F_plus, region_ids, R, {},
                          history, t, evaluate_fn, xl, xu, budget,
                          base_seed, event_id)
    Q_keep = quality_fn(keep_result["F"])

    active_regions = [r for r in range(R) if np.any(region_ids == r)]

    utilities = {}
    preferred_action = {}
    for r in active_regions:
        utilities[r] = {"KEEP": 0.0}
        for action in ("LOCAL", "PREDICT", "DIVERSIFY"):
            result = respond(X_base, F_plus, region_ids, R, {r: action},
                             history, t, evaluate_fn, xl, xu, budget,
                             base_seed, event_id)
            Q = quality_fn(result["F"])
            utilities[r][action] = float(Q - Q_keep)

        best_action = "KEEP"
        best_u = 0.0
        for action in ACTIONS:
            u = utilities[r].get(action, float("nan"))
            if np.isnan(u):
                continue
            if u > best_u:      # strict > -- tie voi KEEP thi KEEP thang
                best_u = u
                best_action = action
        preferred_action[r] = best_action

    OHI = operator_heterogeneity_index(
        [preferred_action[r] for r in active_regions])

    # -------------------- best global action --------------------
    global_results = {}
    for action in ACTIONS:
        if action == "KEEP":
            global_results[action] = {"available": True, "Q": float(Q_keep),
                                      "IGD": float(-Q_keep), "extra_fe": 0}
            continue
        actions_map = {r: action for r in active_regions}
        result = respond(X_base, F_plus, region_ids, R, actions_map,
                         history, t, evaluate_fn, xl, xu, budget,
                         base_seed, event_id)
        Q = quality_fn(result["F"])
        global_results[action] = {"available": True, "Q": float(Q),
                                  "IGD": float(-Q),
                                  "extra_fe": int(result["extra_fe"])}

    best_global_action = None
    best_global_Q = -np.inf
    for action in ACTIONS:
        info = global_results[action]
        if info["Q"] > best_global_Q:
            best_global_Q = info["Q"]
            best_global_action = action

    # -------------------- regional oracle (combined) --------------------
    regional_actions = {r: preferred_action[r] for r in active_regions}
    regional_result = respond(X_base, F_plus, region_ids, R,
                              regional_actions, history, t,
                              evaluate_fn, xl, xu, budget, base_seed,
                              event_id)
    Q_regional = quality_fn(regional_result["F"])

    eps = 1e-12
    ROG = (Q_regional - best_global_Q) / (abs(best_global_Q) + eps)
    Delta_Q = Q_regional - best_global_Q
    Delta_IGD = (-best_global_Q) - (-Q_regional)

    return {
        "Q_keep": float(Q_keep), "IGD_keep": float(-Q_keep),
        "utilities": utilities, "preferred_action": preferred_action,
        "OHI": float(OHI),
        "global_results": global_results,
        "best_global_action": best_global_action,
        "best_global_Q": float(best_global_Q),
        "best_global_IGD": float(-best_global_Q),
        "regional_actions": regional_actions,
        "regional_Q": float(Q_regional), "regional_IGD": float(-Q_regional),
        "regional_extra_fe": int(regional_result["extra_fe"]),
        "Delta_Q": float(Delta_Q), "Delta_IGD": float(Delta_IGD),
        "ROG": float(ROG),
    }


# ============================================================
# One full run: warm-up -> baseline trajectory -> scored events
# (Section 8: oracle KHONG anh huong baseline trajectory).
# ============================================================

def run_one(scenario, seed, n_scored_changes=5, n_bins=None):
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R = len(V)
    n_bins = R if n_bins is None else n_bins
    region_dynamics = make_region_dynamics_map(scenario, n_bins)
    pf_star = get_true_pf(n_points=500)

    rng = np.random.default_rng(seed)
    X_pop = rng.uniform(xl, xu, size=(POP_SIZE, D))

    t_current = compute_time(0)
    X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                         WARMUP_GENERATIONS, seed * 10000, xl, xu)

    # history[k] = snapshot của X_pop ngay trước change_idx=k+1.
    # PREDICT tại change_idx=c đọc history[c-2] và history[c-1].
    history = RegionHistory(n_regions=R)

    rows = []
    gen = 0

    n_total_changes = 1 + n_scored_changes  # change_idx=1 = history-only

    for change_idx in range(1, n_total_changes + 1):
        F_minus = op_evaluate(X_pop, t_current, region_dynamics)
        gen += GENERATIONS_PER_CHANGE
        t_new = compute_time(gen)
        F_plus = op_evaluate(X_pop, t_new, region_dynamics)

        F_tilde_minus, F_tilde_plus = normalize(F_minus, F_plus)
        F_tilde_minus = np.asarray(F_tilde_minus)
        F_tilde_plus = np.asarray(F_tilde_plus)

        region_ids = np.asarray(partition(F_tilde_minus, V))

        # Lưu snapshot hiện tại vào history TRƯỚC khi chạy oracle.
        # history[change_idx-1] = X_pop với partition tại thời điểm này.
        # Oracle tại change_idx dùng t=change_idx -> đọc history[change_idx-2]
        # và history[change_idx-1].  Không có future leakage: chỉ dùng
        # F_minus (old env) và X_pop chưa có thông tin từ oracle.
        history.update(
            t=change_idx - 1,
            X=X_pop,
            F=F_minus,
            region_ids=region_ids,
            xl=xl,
            xu=xu,
        )

        # Chi de GHI (occupancy/severity diagnostic) -- KHONG dung de
        # chon action o bat ky dau (Section 25).
        s_r, c_r, _q = severity(F_tilde_minus, F_tilde_plus, region_ids,
                                R, q_prev=1.0)

        if change_idx == 1:
            # Event đầu tiên: chỉ xây history, KHÔNG scored.
            t_current = t_new
            X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                                 GENERATIONS_PER_CHANGE,
                                 seed * 10000 + change_idx, xl, xu)
            continue

        def evaluate_fn(X_new, _t=t_new, _rd=region_dynamics):
            return op_evaluate(X_new, _t, _rd)

        oracle_result = run_oracle_event(
            X_base=X_pop, F_plus=F_plus, region_ids=region_ids, R=R,
            history=history, t=change_idx,
            evaluate_fn=evaluate_fn,
            pf_star=pf_star, xl=xl, xu=xu, budget=BUDGET,
            base_seed=seed, event_id=change_idx)

        active_mask = ~np.isnan(np.asarray(s_r, dtype=float))
        R_active = int(np.sum(active_mask))

        row = {
            "scenario": scenario, "seed": seed, "change_idx": change_idx,
            "t_old": float(t_current), "t_new": float(t_new),
            "R_total": R, "R_active": R_active,
            "active_fraction": R_active / R,
            "region_dynamics": region_dynamics,
            "severity": np.asarray(s_r, dtype=float).tolist(),
            "Q_keep": oracle_result["Q_keep"],
            "IGD_keep": oracle_result["IGD_keep"],
            "utilities": oracle_result["utilities"],
            "preferred_action": oracle_result["preferred_action"],
            "OHI": oracle_result["OHI"],
            "global_results": oracle_result["global_results"],
            "best_global_action": oracle_result["best_global_action"],
            "best_global_Q": oracle_result["best_global_Q"],
            "best_global_IGD": oracle_result["best_global_IGD"],
            "regional_actions": oracle_result["regional_actions"],
            "regional_Q": oracle_result["regional_Q"],
            "regional_IGD": oracle_result["regional_IGD"],
            "regional_extra_fe": oracle_result["regional_extra_fe"],
            "Delta_Q": oracle_result["Delta_Q"],
            "Delta_IGD": oracle_result["Delta_IGD"],
            "ROG": oracle_result["ROG"],
        }
        rows.append(row)

        # -------------------- advance baseline, DISCARD oracle pops --------
        t_current = t_new
        X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                             GENERATIONS_PER_CHANGE,
                             seed * 10000 + change_idx, xl, xu)

    return rows


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def summarize_scenario(scenario, rows):
    if not rows:
        return None
    ohi = np.array([r["OHI"] for r in rows])
    rog = np.array([r["ROG"] for r in rows])
    digd = np.array([r["Delta_IGD"] for r in rows])
    fe = np.array([r["regional_extra_fe"] for r in rows])

    def iqr(a):
        return float(np.percentile(a, 75) - np.percentile(a, 25))

    best_global_counts = {}
    preferred_counts = {}
    for r in rows:
        best_global_counts[r["best_global_action"]] = (
            best_global_counts.get(r["best_global_action"], 0) + 1)
        for a in r["preferred_action"].values():
            preferred_counts[a] = preferred_counts.get(a, 0) + 1

    return {
        "scenario": scenario, "n_events": len(rows),
        "median_OHI": float(np.median(ohi)), "OHI_iqr": iqr(ohi),
        "frac_OHI_gt_0": float(np.mean(ohi > 0)),
        "median_ROG": float(np.median(rog)), "ROG_iqr": iqr(rog),
        "frac_ROG_gt_0": float(np.mean(rog > 0)),
        "median_Delta_IGD": float(np.median(digd)),
        "median_regional_extra_fe": float(np.median(fe)),
        "best_global_action_distribution": best_global_counts,
        "preferred_action_distribution": preferred_counts,
    }


def run_e2a(seeds, n_scored_changes, out_dir=None, label=""):
    print("=" * 70)
    print(f"E2a — Controlled Regional Operator Oracle {label}")
    print(f"seeds={list(seeds)}  n_scored_changes={n_scored_changes}")
    print("=" * 70)

    t0 = time.time()
    all_rows = []
    summaries = {}
    for scenario in SCENARIOS:
        print(f"\n{scenario} ...", end=" ", flush=True)
        scenario_rows = []
        for seed in seeds:
            rows = run_one(scenario, seed, n_scored_changes=n_scored_changes)
            scenario_rows.extend(rows)
        all_rows.extend(scenario_rows)
        summ = summarize_scenario(scenario, scenario_rows)
        summaries[scenario] = summ
        print(f"n={summ['n_events']}  median_OHI={summ['median_OHI']:.3f}  "
              f"median_ROG={summ['median_ROG']:.3f}  "
              f"frac_ROG>0={summ['frac_ROG_gt_0']:.1%}")

    elapsed = time.time() - t0
    print(f"\nTime: {elapsed:.1f}s")
    print("=" * 70)

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        jsonl_path = out_path / "E2a_controlled_operator_oracle.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in all_rows:
                f.write(json.dumps(json_safe(r), ensure_ascii=False) + "\n")

        csv_path = out_path / "E2a_controlled_operator_oracle_summary.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            cols = ["scenario", "n_events", "median_OHI", "OHI_iqr",
                    "frac_OHI_gt_0", "median_ROG", "ROG_iqr",
                    "frac_ROG_gt_0", "median_Delta_IGD",
                    "median_regional_extra_fe"]
            f.write(",".join(cols) + "\n")
            for scenario in SCENARIOS:
                s = summaries[scenario]
                f.write(",".join(str(s[c]) for c in cols) + "\n")

        report_path = out_path / "E2a_controlled_operator_oracle_report.md"
        with report_path.open("w", encoding="utf-8") as f:
            f.write("# E2a — Controlled Regional Operator Oracle\n\n")
            for scenario in SCENARIOS:
                s = summaries[scenario]
                f.write(f"## {scenario}\n\n")
                f.write(f"- events: {s['n_events']}\n")
                f.write(f"- median OHI: {s['median_OHI']:.4f} "
                        f"(IQR {s['OHI_iqr']:.4f}), "
                        f"frac OHI>0: {s['frac_OHI_gt_0']:.1%}\n")
                f.write(f"- median ROG: {s['median_ROG']:.4f} "
                        f"(IQR {s['ROG_iqr']:.4f}), "
                        f"frac ROG>0: {s['frac_ROG_gt_0']:.1%}\n")
                f.write(f"- median Delta_IGD: {s['median_Delta_IGD']:.6f}\n")
                f.write(f"- median regional extra FE: "
                        f"{s['median_regional_extra_fe']:.1f}\n")
                f.write(f"- best-global action distribution: "
                        f"{s['best_global_action_distribution']}\n")
                f.write(f"- preferred regional action distribution: "
                        f"{s['preferred_action_distribution']}\n\n")

        print(f"\nSaved: {jsonl_path}, {csv_path}, {report_path}")

    return {"rows": all_rows, "summaries": summaries}


if __name__ == "__main__":
    run_e2a(seeds=[1, 2, 3], n_scored_changes=5, label="(smoke)")
