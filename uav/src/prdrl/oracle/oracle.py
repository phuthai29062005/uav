import numpy as np

from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from prdrl.operators.local import local
from prdrl.operators.predict import predict
from prdrl.operators.diversify import diversify
from prdrl.operators.budget import equal_budget

from prdrl.features.forecast_state import (
    get_predict_motion_from_history,
)

# D2-A: canonical HARD survival helper (also imported by R1 diagnostic).
# Import is deferred to avoid circular dependencies — loaded on first use
# via the lazy accessor below.
_region_protected_survival = None


def _get_region_protected_survival():
    global _region_protected_survival
    if _region_protected_survival is None:
        from prdrl.core.survival import region_protected_survival
        _region_protected_survival = region_protected_survival
    return _region_protected_survival


# Predict response survival modes (D2-A).
PREDICT_SURVIVAL_UNRESTRICTED = "UNRESTRICTED"
PREDICT_SURVIVAL_HARD_RESPONSE_ONLY = "HARD_RESPONSE_ONLY"


ACTIONS = (
    "KEEP",
    "LOCAL",
    "PREDICT",
    "DIVERSIFY",
)


# ============================================================
# 1. Chia X thành các region
# ============================================================

def split_by_region(X, region_ids, R):

    X = np.asarray(X, dtype=float)
    region_ids = np.asarray(region_ids)

    regions = {}

    for r in range(R):
        regions[r] = X[region_ids == r].copy()

    return regions


# ============================================================
# 2. RNG riêng cho từng event / region / action
#
# Mục đích:
# cùng một region + action tại cùng một event phải sinh cùng
# random stream dù được gọi trong isolated/global/regional oracle.
#
# Có t trong seed để tránh lặp lại cùng noise ở mọi environment.
# ============================================================

def make_rng(
    base_seed,
    t,
    region_id,
    action,
):

    action_code = {
        "KEEP": 0,
        "LOCAL": 1,
        "PREDICT": 2,
        "DIVERSIFY": 3,
        "RESPONSE_SURVIVAL": 4,
    }

    if action not in action_code:
        raise ValueError(f"Unknown action: {action}")

    seed_sequence = np.random.SeedSequence([
        int(base_seed),
        int(t),
        int(region_id),
        action_code[action],
    ])

    return np.random.default_rng(seed_sequence)


# ============================================================
# 3. NSGA-II survival
#
# Parent cũ + response offspring
#             ↓
#       giữ lại đúng N nghiệm
# ============================================================

def nsga2_survival(
    X,
    F,
    n_survive,
    xl,
    xu,
    response_rng=None,
):

    X = np.asarray(X, dtype=float)
    F = np.asarray(F, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if F.ndim != 2:
        raise ValueError("F must be a 2D array.")

    if X.shape[0] != F.shape[0]:
        raise ValueError("X and F must contain the same number of solutions.")

    D = X.shape[1]
    M = F.shape[1]

    problem = Problem(
        n_var=D,
        n_obj=M,
        xl=xl,
        xu=xu,
    )

    pop = Population.new(
        X=X,
        F=F,
    )

    survival = RankAndCrowding()

    survived = survival.do(
        problem,
        pop,
        n_survive=n_survive,
        random_state=response_rng,
    )

    X_new = survived.get("X")
    F_new = survived.get("F")

    return X_new, F_new


# ============================================================
# 4. Sinh offspring cho MỘT region
#
# PREDICT hiện tại:
#
# mu_{t-2}, mu_{t-1}
#       ↓
# delta_hat = lambda * (mu_{t-1} - mu_{t-2})
#       ↓
# predict()
#
# Nếu thiếu history:
# delta_hat = None
#       ↓
# predict() fallback sang LOCAL với sigma_local
# ============================================================

def generate_children(
    action,
    region_id,
    X_region,
    history,
    t,
    budget,
    xl,
    xu,
    sigma_local,
    sigma_predict,
    sigma_diversify,
    lambda_pred,
    base_seed,
):

    X_region = np.asarray(
        X_region,
        dtype=float,
    )

    xl = np.asarray(
        xl,
        dtype=float,
    )

    xu = np.asarray(
        xu,
        dtype=float,
    )

    D = len(xl)

    # --------------------------------------------------------
    # Region rỗng
    # --------------------------------------------------------

    if len(X_region) == 0:
        return np.empty(
            (0, D),
            dtype=float,
        )

    # --------------------------------------------------------
    # KEEP không sinh response offspring
    # --------------------------------------------------------

    if action == "KEEP":
        return np.empty(
            (0, D),
            dtype=float,
        )

    # --------------------------------------------------------
    # Budget
    # --------------------------------------------------------

    n_offspring = equal_budget(
        action=action,
        region_size=len(X_region),
        budget=budget,
    )

    if n_offspring == 0:
        return np.empty(
            (0, D),
            dtype=float,
        )

    # --------------------------------------------------------
    # RNG deterministic theo:
    # event / region / action
    # --------------------------------------------------------

    rng = make_rng(
        base_seed=base_seed,
        t=t,
        region_id=region_id,
        action=action,
    )

    # --------------------------------------------------------
    # LOCAL
    # --------------------------------------------------------

    if action == "LOCAL":

        return local(
            X_region=X_region,
            n_offspring=n_offspring,
            xl=xl,
            xu=xu,
            sigma=sigma_local,
            rng=rng,
        )

    # --------------------------------------------------------
    # DIVERSIFY
    # --------------------------------------------------------

    if action == "DIVERSIFY":

        return diversify(
            X_region=X_region,
            n_offspring=n_offspring,
            xl=xl,
            xu=xu,
            sigma=sigma_diversify,
            rng=rng,
        )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if action == "PREDICT":

        motion = get_predict_motion_from_history(
            history=history,
            region_id=region_id,
            t=t,
            lambda_pred=lambda_pred,
        )

        # Không đủ:
        # mu^{t-2}, mu^{t-1}
        #
        # -> predict() sẽ fallback sang LOCAL.
        if motion is None:
            delta_hat = None
        else:
            delta_hat = motion["delta_hat"]

        return predict(
            X_prev=X_region,
            delta_hat=delta_hat,
            n_offspring=n_offspring,
            xl=xl,
            xu=xu,
            sigma_pred=sigma_predict,
            rng=rng,
            sigma_local=sigma_local,
        )

    raise ValueError(
        f"Unknown action: {action}"
    )


def generate_children_with_info(
    action,
    region_id,
    X_region,
    old_region_ids,
    history,
    t,
    budget,
    xl,
    xu,
    sigma_local,
    sigma_predict,
    sigma_diversify,
    lambda_pred,
    base_seed,
):
    """Wrapper around generate_children that also returns metadata needed for HARD survival.

    Returns:
        children (B, D)   — generated children (may be empty)
        realized_op (str) — actual operator realized:
                            "PREDICT"        successful predict (has valid delta_hat)
                            "LOCAL_FALLBACK" predict requested but fell back to LOCAL
                            "LOCAL"          LOCAL action directly
                            "DIVERSIFY"      DIVERSIFY action
                            "KEEP"           KEEP action (no children)
        n_r_before (int)  — number of OLD individuals assigned to region_id before response
    """
    old_region_ids = np.asarray(old_region_ids)
    n_r_before = int(np.sum(old_region_ids == region_id))

    # Determine realized operator before calling generate_children
    if action == "PREDICT":
        motion = get_predict_motion_from_history(
            history=history,
            region_id=region_id,
            t=t,
            lambda_pred=lambda_pred,
        )
        realized_op = "PREDICT" if motion is not None else "LOCAL_FALLBACK"
    elif action in ("LOCAL", "DIVERSIFY", "KEEP"):
        realized_op = action
    else:
        realized_op = action

    children = generate_children(
        action=action,
        region_id=region_id,
        X_region=X_region,
        history=history,
        t=t,
        budget=budget,
        xl=xl,
        xu=xu,
        sigma_local=sigma_local,
        sigma_predict=sigma_predict,
        sigma_diversify=sigma_diversify,
        lambda_pred=lambda_pred,
        base_seed=base_seed,
    )

    return children, realized_op, n_r_before


# ============================================================
# 5. Apply một action-map lên toàn population
#
# Ví dụ:
#
# {
#     0: "KEEP",
#     1: "LOCAL",
#     2: "KEEP",
#     3: "DIVERSIFY",
# }
#
# ============================================================

def respond_population(
    X,
    F_plus,
    region_ids,
    V,
    actions,
    history,
    t,
    evaluate_fn,
    xl,
    xu,
    budget,
    sigma_local,
    sigma_predict,
    sigma_diversify,
    lambda_pred,
    base_seed,
    predict_response_survival_mode=PREDICT_SURVIVAL_UNRESTRICTED,
):

    X = np.asarray(
        X,
        dtype=float,
    )

    F_plus = np.asarray(
        F_plus,
        dtype=float,
    )

    region_ids = np.asarray(
        region_ids,
    )

    R = len(V)
    N = len(X)

    use_hard = (predict_response_survival_mode == PREDICT_SURVIVAL_HARD_RESPONSE_ONLY)

    children_all = []

    # Per-region metadata collected when HARD mode is active.
    # List of (region_id, realized_op, n_r_before, child_start_idx, n_children).
    region_meta = []
    child_offset = 0

    # ========================================================
    # Sinh offspring từng region
    # ========================================================

    for r in range(R):

        mask = region_ids == r

        X_region = X[mask]

        if len(X_region) == 0:
            continue

        action = actions.get(
            r,
            "KEEP",
        )

        if use_hard:
            children, realized_op, n_r_before = generate_children_with_info(
                action=action,
                region_id=r,
                X_region=X_region,
                old_region_ids=region_ids,
                history=history,
                t=t,
                budget=budget,
                xl=xl,
                xu=xu,
                sigma_local=sigma_local,
                sigma_predict=sigma_predict,
                sigma_diversify=sigma_diversify,
                lambda_pred=lambda_pred,
                base_seed=base_seed,
            )
            n_ch = len(children)
            region_meta.append((r, realized_op, n_r_before, child_offset, n_ch))
            child_offset += n_ch
        else:
            children = generate_children(
                action=action,
                region_id=r,
                X_region=X_region,
                history=history,
                t=t,
                budget=budget,
                xl=xl,
                xu=xu,
                sigma_local=sigma_local,
                sigma_predict=sigma_predict,
                sigma_diversify=sigma_diversify,
                lambda_pred=lambda_pred,
                base_seed=base_seed,
            )

        if len(children) > 0:
            children_all.append(
                children
            )

    # ========================================================
    # Không có response offspring
    #
    # Ví dụ toàn bộ KEEP
    # ========================================================

    if len(children_all) == 0:

        return {
            "available": True,
            "X": X.copy(),
            "F": F_plus.copy(),
            "extra_fe": 0,
        }

    # ========================================================
    # Ghép response children
    # ========================================================

    X_children = np.vstack(
        children_all
    )

    # ========================================================
    # Evaluate children
    #
    # Đây chính là extra response FE.
    # ========================================================

    F_children = evaluate_fn(
        X_children
    )

    F_children = np.asarray(
        F_children,
        dtype=float,
    )

    extra_fe = len(
        X_children
    )

    # ========================================================
    # Survival về đúng N nghiệm
    # ========================================================

    # Deterministic RNG for response survival (D2-RF: namespace = seed/change/region).
    _active_regions = sorted(r for r, a in actions.items() if a != "KEEP")
    _surv_region = _active_regions[0] if _active_regions else 0
    surv_rng = make_rng(base_seed, t, _surv_region, "RESPONSE_SURVIVAL")

    if use_hard:
        # Determine if exactly one region realized a successful PREDICT.
        predict_regions = [
            (r, n_r_before)
            for (r, realized_op, n_r_before, cs, nc) in region_meta
            if realized_op == "PREDICT"
        ]

        if len(predict_regions) == 1:
            # HARD path: single successful PREDICT target.
            target_r, n_r_before = predict_regions[0]
            hard_surv = _get_region_protected_survival()
            X_new, F_new, _ = hard_surv(
                X_pop=X,
                F_pop=F_plus,
                X_children=X_children,
                F_children=F_children,
                response_region_r=target_r,
                old_region_ids=region_ids,
                n_r_before=n_r_before,
                xl=xl,
                xu=xu,
            )
        else:
            # LOCAL_FALLBACK or multi-PREDICT: unrestricted survival.
            X_pool = np.vstack([X, X_children])
            F_pool = np.vstack([F_plus, F_children])
            X_new, F_new = nsga2_survival(
                X=X_pool,
                F=F_pool,
                n_survive=N,
                xl=xl,
                xu=xu,
                response_rng=surv_rng,
            )
    else:
        # O0 default: unrestricted NSGA-II.
        X_pool = np.vstack([
            X,
            X_children,
        ])

        F_pool = np.vstack([
            F_plus,
            F_children,
        ])

        X_new, F_new = nsga2_survival(
            X=X_pool,
            F=F_pool,
            n_survive=N,
            xl=xl,
            xu=xu,
            response_rng=surv_rng,
        )

    return {
        "available": True,
        "X": X_new,
        "F": F_new,
        "extra_fe": int(extra_fe),
    }


# ============================================================
# 6. Operator Heterogeneity Index
#
# OHI =
#
#     - sum_a p_a log(p_a)
#     --------------------
#          log |A|
#
# ============================================================

def operator_heterogeneity_index(
    preferred_actions,
    eps=1e-12,
):

    if len(preferred_actions) < 2:
        return 0.0

    counts = []

    for action in ACTIONS:

        count = sum(
            a == action
            for a in preferred_actions
        )

        counts.append(
            count
        )

    counts = np.asarray(
        counts,
        dtype=float,
    )

    if counts.sum() == 0:
        return 0.0

    p = counts / counts.sum()

    nonzero = p > 0

    entropy = -np.sum(
        p[nonzero]
        * np.log(
            p[nonzero] + eps
        )
    )

    return float(
        entropy
        / np.log(len(ACTIONS))
    )


# ============================================================
# 7. REGIONAL OPERATOR ORACLE
#
# Utility:
#
# U_{r,a}
# =
# Q(P^(r,a)) - Q(P^KEEP)
#
#
# Preferred action:
#
# a*_r = argmax_a U_{r,a}
#
#
# ROG:
#
# Q(regional) - Q(best global)
# ----------------------------
# |Q(best global)| + eps
#
# ============================================================

def regional_oracle(
    X,
    F_plus,
    region_ids,
    V,
    history,
    t,
    evaluate_fn,
    quality_fn,
    xl,
    xu,
    budget=10,
    sigma_local=0.02,
    sigma_predict=0.005,  # O1 adoption: D2-D 2026-09-20 (was 0.03)
    sigma_diversify=0.15,
    lambda_pred=1.0,
    base_seed=42,
    eps=1e-12,
):

    X = np.asarray(
        X,
        dtype=float,
    )

    F_plus = np.asarray(
        F_plus,
        dtype=float,
    )

    region_ids = np.asarray(
        region_ids,
    )

    R = len(V)

    # ========================================================
    # A. KEEP baseline
    # ========================================================

    keep_result = respond_population(
        X=X,
        F_plus=F_plus,
        region_ids=region_ids,
        V=V,
        actions={},
        history=history,
        t=t,
        evaluate_fn=evaluate_fn,
        xl=xl,
        xu=xu,
        budget=budget,
        sigma_local=sigma_local,
        sigma_predict=sigma_predict,
        sigma_diversify=sigma_diversify,
        lambda_pred=lambda_pred,
        base_seed=base_seed,
    )

    Q_keep = quality_fn(
        keep_result["F"]
    )

    # ========================================================
    # B. Isolated regional trials
    #
    # Chỉ region r dùng action a.
    # Mọi region khác KEEP.
    # ========================================================

    utilities = {}
    preferred_action = {}
    active_regions = []

    for r in range(R):

        region_size = np.sum(
            region_ids == r
        )

        if region_size == 0:
            continue

        active_regions.append(
            r
        )

        # KEEP utility luôn = 0
        utilities[r] = {
            "KEEP": 0.0
        }

        # ----------------------------------------------------
        # Thử từng active response operator
        # ----------------------------------------------------

        for action in (
            "LOCAL",
            "PREDICT",
            "DIVERSIFY",
        ):

            actions = {
                r: action
            }

            result = respond_population(
                X=X,
                F_plus=F_plus,
                region_ids=region_ids,
                V=V,
                actions=actions,
                history=history,
                t=t,
                evaluate_fn=evaluate_fn,
                xl=xl,
                xu=xu,
                budget=budget,
                sigma_local=sigma_local,
                sigma_predict=sigma_predict,
                sigma_diversify=sigma_diversify,
                lambda_pred=lambda_pred,
                base_seed=base_seed,
            )

            Q_action = quality_fn(
                result["F"]
            )

            # -----------------------------------------------
            # U_{r,a}
            # -----------------------------------------------

            U = (
                Q_action
                - Q_keep
            )

            utilities[r][action] = float(
                U
            )

        # ====================================================
        # Preferred action của region r
        #
        # KEEP thắng tie:
        # chỉ đổi action nếu utility > 0 hiện tại.
        # ====================================================

        best_action = "KEEP"
        best_utility = 0.0

        for action in ACTIONS:

            u = utilities[r].get(
                action,
                np.nan,
            )

            if np.isnan(u):
                continue

            if u > best_utility:
                best_utility = u
                best_action = action

        preferred_action[r] = (
            best_action
        )

    # ========================================================
    # C. OHI
    # ========================================================

    preferred_list = [
        preferred_action[r]
        for r in active_regions
    ]

    OHI = operator_heterogeneity_index(
        preferred_list
    )

    # ========================================================
    # D. BEST GLOBAL ACTION
    #
    # Cùng một action áp cho toàn bộ active regions.
    # ========================================================

    global_results = {}

    for action in ACTIONS:

        # ----------------------------------------------------
        # KEEP baseline
        # ----------------------------------------------------

        if action == "KEEP":

            global_results[action] = {
                "available": True,
                "Q": float(Q_keep),
                "extra_fe": 0,
            }

            continue

        # ----------------------------------------------------
        # Cùng một action cho mọi active region
        # ----------------------------------------------------

        actions = {
            r: action
            for r in active_regions
        }

        result = respond_population(
            X=X,
            F_plus=F_plus,
            region_ids=region_ids,
            V=V,
            actions=actions,
            history=history,
            t=t,
            evaluate_fn=evaluate_fn,
            xl=xl,
            xu=xu,
            budget=budget,
            sigma_local=sigma_local,
            sigma_predict=sigma_predict,
            sigma_diversify=sigma_diversify,
            lambda_pred=lambda_pred,
            base_seed=base_seed,
        )

        Q_global = quality_fn(
            result["F"]
        )

        global_results[action] = {
            "available": True,
            "Q": float(Q_global),
            "extra_fe": int(
                result["extra_fe"]
            ),
        }

    # ========================================================
    # Chọn best-global
    #
    # ACTIONS bắt đầu bằng KEEP.
    # Nếu Q tie chính xác, action xuất hiện trước sẽ giữ chiến thắng.
    # ========================================================

    best_global_action = None
    best_global_Q = -np.inf

    for action in ACTIONS:

        info = global_results[
            action
        ]

        Q = info["Q"]

        if Q > best_global_Q:

            best_global_Q = Q
            best_global_action = action

    # ========================================================
    # E. REGIONAL ORACLE
    #
    # Ghép preferred action của tất cả regions rồi chạy thật.
    # ========================================================

    regional_actions = {
        r: preferred_action[r]
        for r in active_regions
    }

    regional_result = respond_population(
        X=X,
        F_plus=F_plus,
        region_ids=region_ids,
        V=V,
        actions=regional_actions,
        history=history,
        t=t,
        evaluate_fn=evaluate_fn,
        xl=xl,
        xu=xu,
        budget=budget,
        sigma_local=sigma_local,
        sigma_predict=sigma_predict,
        sigma_diversify=sigma_diversify,
        lambda_pred=lambda_pred,
        base_seed=base_seed,
    )

    Q_regional = quality_fn(
        regional_result["F"]
    )

    # ========================================================
    # F. REGIONAL ORACLE GAIN
    # ========================================================

    ROG = (
        Q_regional
        - best_global_Q
    ) / (
        abs(best_global_Q)
        + eps
    )

    return {

        "Q_keep":
            float(Q_keep),

        "utilities":
            utilities,

        "preferred_action":
            preferred_action,

        "OHI":
            float(OHI),

        "global_results":
            global_results,

        "best_global_action":
            best_global_action,

        "best_global_Q":
            float(best_global_Q),

        "regional_actions":
            regional_actions,

        "regional_Q":
            float(Q_regional),

        "regional_extra_fe":
            int(
                regional_result["extra_fe"]
            ),

        "ROG":
            float(ROG),
    }