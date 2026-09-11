# Changelog

## 4.5B — propagate problem bounds (CRITICAL)

**Invalidates all prior benchmark results on DF3-DF10, DF12-DF14 (bounds bug).**
Offspring were clipped to [0,1] in nsga2_pymoo.py while 11/14 DF problems
have bounds != [0,1] (DF4 is [-2,2]); every result on those problems was
run on a wrongly constrained search space. DF1, DF2, DF11 ([0,1]) unaffected.

- nsga2_pymoo: clip offspring to problem.xl/xu, not [0,1].
- sa_drl_dmoea: repair, action_local/predict/diversify/memory_global,
  apply_hierarchical_response, compute_entropy now require xl/xu; noise and
  diversify scale by per-dim width; no hard-coded [0,1].
- change_detector: requires lb/ub (no default); perturbation applied in
  physical space as h*d*(ub-lb), derivative taken w.r.t. normalized coord.
- dynamic_runner: population init as xl + rand*(xu-xl); bounds threaded through.

Audit: DF4 static IGD 0.0138 after 200 gens (was ~0.53 clipped to [0,1]).
