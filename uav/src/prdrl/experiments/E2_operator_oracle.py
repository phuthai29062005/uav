import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

from prdrl.core.normalize import normalize
from prdrl.core.reference import reference
from prdrl.core.partition import partition

from prdrl.detectors.severity import severity
from prdrl.detectors.quality_loss import quality_loss

from prdrl.metrics.igd import igd

from prdrl.problems.cec2018 import (
    evaluate_df,
    get_true_pf,
    get_df_bounds,
    get_n_obj,
)

from prdrl.oracle.oracle import (
    regional_oracle,
    split_by_region,
)

from prdrl.history.region_history import RegionHistory


# ============================================================
# CONFIG
# ============================================================

DF_IDS = list(range(1, 15))

SEEDS = [1, 2, 3]

D = 10

WARMUP_GENERATIONS = 50

N_CHANGES = 15

GENERATIONS_PER_CHANGE = 10

N_T = 10

# equal response budget cho E2 oracle
OPERATOR_BUDGET = 10

# operator parameters
SIGMA_LOCAL = 0.02

SIGMA_PREDICT = 0.03

SIGMA_DIVERSIFY = 0.15

LAMBDA_PREDICT = 1.0

THETA_PBI = 5.0


# ============================================================
# Dynamic problem wrapper
# ============================================================

class DFProblem(Problem):

    def __init__(
        self,
        df_id,
        t,
        D,
        xl,
        xu,
        n_obj,
    ):

        super().__init__(
            n_var=D,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
        )

        self.df_id = df_id
        self.t = t

    def _evaluate(
        self,
        X,
        out,
        *args,
        **kwargs,
    ):

        out["F"] = evaluate_df(
            self.df_id,
            X,
            self.t,
        )


# ============================================================
# Population size
# ============================================================

def get_pop_size(n_obj):

    if n_obj == 2:
        return 100

    if n_obj == 3:
        return 150

    raise ValueError(
        f"Unsupported n_obj={n_obj}"
    )


# ============================================================
# Reference directions
#
# n_partitions != số regions.
#
# 2 objectives:
#     n_partitions=9
#     => R=10
#
# 3 objectives:
#     n_partitions=4
#     => R=15
# ============================================================

def make_reference_directions(n_obj):

    if n_obj == 2:
        n_partitions = 9

    elif n_obj == 3:
        n_partitions = 4

    else:
        raise ValueError(
            f"Unsupported n_obj={n_obj}"
        )

    V = reference(
        n_obj,
        n_partitions
    )

    return V


# ============================================================
# Khởi tạo population đúng bounds
# ============================================================

def initialize_population(
    pop_size,
    xl,
    xu,
    rng,
):

    return rng.uniform(
        low=xl,
        high=xu,
        size=(
            pop_size,
            len(xl),
        ),
    )


# ============================================================
# Evolve NSGA-II từ population hiện tại
#
# QUAN TRỌNG:
# dùng sampling=X_start
# để population hiện tại trở thành initial population.
#
# Không dùng res.X làm population kế tiếp,
# vì res.X thường chỉ là nondominated result,
# số lượng có thể < pop_size.
#
# Dùng:
#     res.pop.get("X")
# ============================================================

def evolve_nsga2(
    df_id,
    t,
    X_start,
    n_gen,
    seed,
    xl,
    xu,
    n_obj,
):

    pop_size = len(X_start)

    problem = DFProblem(
        df_id=df_id,
        t=t,
        D=X_start.shape[1],
        xl=xl,
        xu=xu,
        n_obj=n_obj,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=X_start,
    )

    result = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )

    X_new = np.asarray(
        result.pop.get("X"),
        dtype=float,
    )

    return X_new


# ============================================================
# Helper convert numpy -> JSON-safe
# ============================================================

def json_safe(obj):

    if isinstance(obj, dict):

        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):

        return [
            json_safe(v)
            for v in obj
        ]

    if isinstance(obj, tuple):

        return [
            json_safe(v)
            for v in obj
        ]

    if isinstance(obj, np.ndarray):

        return obj.tolist()

    if isinstance(
        obj,
        (
            np.integer,
        ),
    ):

        return int(obj)

    if isinstance(
        obj,
        (
            np.floating,
        ),
    ):

        value = float(obj)

        if np.isnan(value):
            return None

        return value

    if isinstance(obj, float):

        if np.isnan(obj):
            return None

        return obj

    return obj


# ============================================================
# E2 cho MỘT DF + MỘT seed
# ============================================================

def run_one(
    df_id,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    n_obj = get_n_obj(
        df_id
    )

    pop_size = get_pop_size(
        n_obj
    )

    xl, xu = get_df_bounds(
        df_id,
        D=D,
    )

    V = make_reference_directions(
        n_obj
    )

    R = len(V)

    # --------------------------------------------------------
    # 1. Initial population
    # --------------------------------------------------------

    X_pop = initialize_population(
        pop_size=pop_size,
        xl=xl,
        xu=xu,
        rng=rng,
    )

    # --------------------------------------------------------
    # 2. Warm-up tại t=0
    # --------------------------------------------------------

    t_current = 0.0

    X_pop = evolve_nsga2(
        df_id=df_id,
        t=t_current,
        X_start=X_pop,
        n_gen=WARMUP_GENERATIONS,
        seed=seed * 10000,
        xl=xl,
        xu=xu,
        n_obj=n_obj,
    )

    # --------------------------------------------------------
    # History của region
    # --------------------------------------------------------

    history = RegionHistory(n_regions=R)

    # causal scale cho severity
    q_prev = 1.0

    rows = []

    # ========================================================
    # CHANGE LOOP
    # ========================================================

    for change_idx in range(
        1,
        N_CHANGES + 1,
    ):

        # ====================================================
        # 3. Population ngay TRƯỚC change
        # ====================================================

        F_minus = evaluate_df(
            df_id,
            X_pop,
            t_current,
        )

        # ====================================================
        # 4. Environment mới
        # ====================================================

        t_new = (
            change_idx
            / N_T
        )

        # ====================================================
        # 5. Re-evaluate CÙNG X ở environment mới
        # ====================================================

        F_plus = evaluate_df(
            df_id,
            X_pop,
            t_new,
        )

        # ====================================================
        # 6. Normalize F- và F+ trên CÙNG scale
        # ====================================================

        (
            F_tilde_minus,
            F_tilde_plus,
        ) = normalize(
            F_minus,
            F_plus,
        )

        # ====================================================
        # 7. Partition CHỈ dùng F_minus
        # ====================================================

        region_ids = partition(
            F_tilde_minus,
            V,
        )

        # ====================================================
        # 8. Update region history (before oracle)
        # ====================================================

        history.update(
            t=change_idx - 1,
            X=X_pop,
            F=F_minus,
            region_ids=region_ids,
            xl=xl,
            xu=xu,
        )

        # ====================================================
        # 9. Severity
        # ====================================================

        (
            s_r,
            c_r,
            q_prev,
        ) = severity(
            F_tilde_minus,
            F_tilde_plus,
            region_ids,
            R,
            q_prev,
        )

        # ====================================================
        # 10. Quality loss
        # ====================================================

        loss_r = quality_loss(
            F_tilde_minus,
            F_tilde_plus,
            region_ids,
            V,
            theta=THETA_PBI,
        )

        # ====================================================
        # 11. True PF cho objective quality oracle
        # ====================================================

        pf = get_true_pf(
            df_id,
            t_new,
            n_points=500,
        )

        if pf is None:

            print(
                f"\nWARNING: "
                f"DF{df_id} "
                f"t={t_new:.2f} "
                f"không lấy được true PF. "
                f"Skip event."
            )

            t_current = t_new

            X_pop = evolve_nsga2(
                df_id=df_id,
                t=t_current,
                X_start=X_pop,
                n_gen=GENERATIONS_PER_CHANGE,
                seed=(
                    seed * 10000
                    + change_idx
                ),
                xl=xl,
                xu=xu,
                n_obj=n_obj,
            )

            continue

        # ====================================================
        # 12. evaluate_fn
        #
        # Oracle offspring luôn được evaluate tại t_new
        # ====================================================

        def evaluate_new(
            X_new,
        ):

            return evaluate_df(
                df_id,
                X_new,
                t_new,
            )

        # ====================================================
        # 13. quality_fn
        #
        # IGD nhỏ hơn = tốt hơn
        #
        # Oracle muốn Q lớn hơn = tốt hơn
        #
        # nên:
        #
        # Q = -IGD
        # ====================================================

        def quality(
            F,
        ):

            return -igd(
                F,
                pf,
            )

        # ====================================================
        # 14. RUN REGIONAL OPERATOR ORACLE
        # ====================================================

        oracle_result = regional_oracle(
            X=X_pop,

            F_plus=F_plus,

            region_ids=region_ids,

            V=V,

            history=history,

            t=change_idx,

            evaluate_fn=
                evaluate_new,

            quality_fn=
                quality,

            xl=xl,
            xu=xu,

            budget=
                OPERATOR_BUDGET,

            sigma_local=
                SIGMA_LOCAL,

            sigma_predict=
                SIGMA_PREDICT,

            sigma_diversify=
                SIGMA_DIVERSIFY,

            lambda_pred=
                LAMBDA_PREDICT,

            base_seed=(
                seed * 100000
                + change_idx
            ),
        )

        # ====================================================
        # 15. Result record
        # ====================================================

        preferred = (
            oracle_result[
                "preferred_action"
            ]
        )

        preferred_counts = {
            "KEEP": 0,
            "LOCAL": 0,
            "PREDICT": 0,
            "DIVERSIFY": 0,
        }

        for action in preferred.values():

            preferred_counts[action] += 1

        # best-global FE
        best_global_action = (
            oracle_result[
                "best_global_action"
            ]
        )

        best_global_info = (
            oracle_result[
                "global_results"
            ][
                best_global_action
            ]
        )

        row = {

            "df":
                df_id,

            "seed":
                seed,

            "change_idx":
                change_idx,

            "t_old":
                float(t_current),

            "t_new":
                float(t_new),

            "n_obj":
                n_obj,

            "R":
                R,

            "active_regions":
                int(
                    len(
                        np.unique(
                            region_ids
                        )
                    )
                ),

            # --------------------------------
            # detector features
            # --------------------------------

            "severity":
                list(s_r),

            "c":
                list(c_r),

            "quality_loss":
                loss_r.tolist(),

            # --------------------------------
            # oracle
            # --------------------------------

            "utilities":
                oracle_result[
                    "utilities"
                ],

            "preferred_action":
                preferred,

            "preferred_KEEP":
                preferred_counts["KEEP"],

            "preferred_LOCAL":
                preferred_counts["LOCAL"],

            "preferred_PREDICT":
                preferred_counts["PREDICT"],

            "preferred_DIVERSIFY":
                preferred_counts[
                    "DIVERSIFY"
                ],

            "OHI":
                oracle_result[
                    "OHI"
                ],

            # --------------------------------
            # keep baseline
            # --------------------------------

            "Q_keep":
                oracle_result[
                    "Q_keep"
                ],

            "IGD_keep":
                -oracle_result[
                    "Q_keep"
                ],

            # --------------------------------
            # best global action
            # --------------------------------

            "best_global_action":
                best_global_action,

            "best_global_Q":
                oracle_result[
                    "best_global_Q"
                ],

            "best_global_IGD":
                -oracle_result[
                    "best_global_Q"
                ],

            "best_global_extra_fe":
                best_global_info[
                    "extra_fe"
                ],

            # --------------------------------
            # regional oracle
            # --------------------------------

            "regional_actions":
                oracle_result[
                    "regional_actions"
                ],

            "regional_Q":
                oracle_result[
                    "regional_Q"
                ],

            "regional_IGD":
                -oracle_result[
                    "regional_Q"
                ],

            "regional_extra_fe":
                oracle_result[
                    "regional_extra_fe"
                ],

            "ROG":
                oracle_result[
                    "ROG"
                ],
        }

        rows.append(
            row
        )

        # ====================================================
        # Console log
        # ====================================================

        print(
            f"\nDF{df_id} "
            f"seed={seed} "
            f"change={change_idx} "
            f"t={t_new:.2f}"
        )

        print(
            f"  OHI = "
            f"{oracle_result['OHI']:.4f}"
        )

        print(
            f"  best global = "
            f"{best_global_action}"
        )

        print(
            f"  regional actions = "
            f"{preferred}"
        )

        print(
            f"  IGD keep = "
            f"{-oracle_result['Q_keep']:.6f}"
        )

        print(
            f"  IGD global = "
            f"{-oracle_result['best_global_Q']:.6f}"
        )

        print(
            f"  IGD regional = "
            f"{-oracle_result['regional_Q']:.6f}"
        )

        print(
            f"  ROG = "
            f"{oracle_result['ROG']:.4f}"
        )

        # ====================================================
        # 17. Advance baseline trajectory
        #
        # CỰC KỲ QUAN TRỌNG:
        #
        # KHÔNG dùng regional_result làm X_pop.
        #
        # Oracle chỉ là counterfactual diagnostic.
        #
        # Baseline trajectory tiếp tục bằng NSGA-II
        # từ population đang có.
        # ====================================================

        t_current = t_new

        X_pop = evolve_nsga2(
            df_id=df_id,
            t=t_current,
            X_start=X_pop,
            n_gen=
                GENERATIONS_PER_CHANGE,
            seed=(
                seed * 10000
                + change_idx
            ),
            xl=xl,
            xu=xu,
            n_obj=n_obj,
        )

    return rows


# ============================================================
# SUMMARY
# ============================================================

def summarize(rows):

    if len(rows) == 0:

        print(
            "Không có E2 rows."
        )

        return

    print()
    print("=" * 80)
    print(
        "E2 — REGIONAL OPERATOR ORACLE SUMMARY"
    )
    print("=" * 80)

    for df_id in DF_IDS:

        df_rows = [
            row
            for row in rows
            if row["df"] == df_id
        ]

        if len(df_rows) == 0:
            continue

        ohi = np.array([
            row["OHI"]
            for row in df_rows
        ])

        rog = np.array([
            row["ROG"]
            for row in df_rows
        ])

        igd_keep = np.array([
            row["IGD_keep"]
            for row in df_rows
        ])

        igd_global = np.array([
            row["best_global_IGD"]
            for row in df_rows
        ])

        igd_regional = np.array([
            row["regional_IGD"]
            for row in df_rows
        ])

        print(
            f"\nDF{df_id}"
        )

        print(
            f"  events = "
            f"{len(df_rows)}"
        )

        print(
            f"  median OHI = "
            f"{np.median(ohi):.4f}"
        )

        print(
            f"  median ROG = "
            f"{np.median(rog):.4f}"
        )

        print(
            f"  ROG > 0 = "
            f"{np.mean(rog > 0):.1%}"
        )

        print(
            f"  median IGD KEEP = "
            f"{np.median(igd_keep):.6f}"
        )

        print(
            f"  median IGD GLOBAL = "
            f"{np.median(igd_global):.6f}"
        )

        print(
            f"  median IGD REGIONAL = "
            f"{np.median(igd_regional):.6f}"
        )

        # --------------------------------
        # preferred action distribution
        # --------------------------------

        action_counts = {
            "KEEP": 0,
            "LOCAL": 0,
            "PREDICT": 0,
            "DIVERSIFY": 0,
        }

        for row in df_rows:

            for action in (
                row[
                    "preferred_action"
                ].values()
            ):

                action_counts[
                    action
                ] += 1

        total_actions = sum(
            action_counts.values()
        )

        print(
            "  preferred actions:"
        )

        for action, count in (
            action_counts.items()
        ):

            if total_actions == 0:
                pct = 0.0
            else:
                pct = (
                    count
                    / total_actions
                )

            print(
                f"    "
                f"{action:<10} "
                f"{count:>5} "
                f"({pct:.1%})"
            )

    # ========================================================
    # Overall
    # ========================================================

    all_rog = np.array([
        row["ROG"]
        for row in rows
    ])

    all_ohi = np.array([
        row["OHI"]
        for row in rows
    ])

    print()
    print("-" * 80)

    print(
        f"ALL EVENTS = "
        f"{len(rows)}"
    )

    print(
        f"Overall median OHI = "
        f"{np.median(all_ohi):.4f}"
    )

    print(
        f"Overall median ROG = "
        f"{np.median(all_rog):.4f}"
    )

    print(
        f"Overall ROG > 0 = "
        f"{np.mean(all_rog > 0):.1%}"
    )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def run_e2():

    print("=" * 80)

    print(
        "E2 — Regional Operator Oracle"
    )

    print(
        "Equal FE budget across "
        "LOCAL / PREDICT / DIVERSIFY"
    )

    print("=" * 80)

    all_rows = []

    for df_id in DF_IDS:

        for seed in SEEDS:

            print()
            print("#" * 80)

            print(
                f"Running "
                f"DF{df_id}, "
                f"seed={seed}"
            )

            print("#" * 80)

            rows = run_one(
                df_id=df_id,
                seed=seed,
            )

            all_rows.extend(
                rows
            )

    # ========================================================
    # Save raw event results
    # ========================================================

    out_dir = Path(
        "results"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        out_dir
        / "E2_operator_oracle.jsonl"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in all_rows:

            safe_row = json_safe(
                row
            )

            f.write(
                json.dumps(
                    safe_row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(
        f"Saved: "
        f"{output_file}"
    )

    summarize(
        all_rows
    )


if __name__ == "__main__":

    run_e2()