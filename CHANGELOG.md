# Changelog

## 4.9 — clean Dynamic NSGA-II baseline

The Dynamic NSGA-II baseline now runs through a dedicated baseline runner
(src/baseline_runner.py::run_nsga2_baseline) and no longer incurs
ChangeDetector, memory, DQN/state-building, or response-operator overhead.
Previously the baseline passed a no-op agent through run_sa_drl, paying for
detector/signature FE and a second (wasted) population reevaluation per
change. Prior baseline runtime/FE comparisons are invalidated. (Prior
NSGA-II quality results were already invalidated by the bounds, tournament,
and timeline fixes in 4.5B/4.5A/4.6A — not solely by 4.9.)

## 4.8 — state semantics cleanup

Default controller state no longer includes CEC oracle phase or the
constant generations-since-change feature (g_norm). 'entropy' renamed to
normalized decision-space dispersion (it was never entropy). State went
from S+6 to S+4: [c_1..c_S, dispersion, hv_drop, d_mem, has_memory].
compute_phase retained in source for a future oracle-phase ablation only;
the live runner does not call it. compute_entropy kept as a deprecated
wrapper for compute_dispersion.

## 4.5B — propagate problem bounds (CRITICAL)

**Invalidates all prior benchmark results on DF3-DF10, DF12-DF14 (bounds bug).**
Offspring were clipped to [0,1] in nsga2_pymoo.py while 11/14 DF problems
have bounds != [0,1] (DF4 is [-2,2]); every result on those problems was
run on a wrongly constrained search space. DF1, DF2, DF11 ([0,1]) are not
affected by the bounds bug, but their results remain preliminary due to
unresolved protocol issues (4.5A canonical NSGA-II, 4.2 reward/horizon,
4.6, 4.7 terminal transition). Not to be read as "valid".

- nsga2_pymoo: clip offspring to problem.xl/xu, not [0,1].
- sa_drl_dmoea: repair, action_local/predict/diversify/memory_global,
  apply_hierarchical_response, compute_entropy now require xl/xu; noise and
  diversify scale by per-dim width; no hard-coded [0,1].
- change_detector: requires lb/ub (no default); perturbation applied in
  physical space as h*d*(ub-lb), derivative taken w.r.t. normalized coord.
- dynamic_runner: population init as xl + rand*(xu-xl); bounds threaded through.

Audit: DF4 static IGD 0.0138 after 200 gens (was ~0.53 clipped to [0,1]).
DF4 static IGD after bounds correction was 0.0138 vs ~0.024 in the
independent audit. Difference may reflect seed, population size, or survival
implementation; deferred to 4.5A canonical-NSGA-II validation.
