# Repository Audit (6.0)

Full inventory before any file is moved. Nothing was deleted. Method
for every file flagged as possibly legacy: `grep -rln` for every plausible
import name across `src/`, `scripts/`, `tests/`; for the root
`dqn_agent.py` also a full `diff` against `src/dqn_agent.py`.

Categories: **A** production core, **B** experiment/training entry
point, **C** analysis/diagnostic, **D** tests, **E** results/checkpoints,
**F** data/benchmark support, **G** documentation, **H**
legacy/duplicate/dead, **I** unknown.

## Root

| path | category | referenced by | duplicate of | status | action |
|---|---|---|---|---|---|
| `dqn_agent.py` | H | *(none — `grep` found zero importers; all `from dqn_agent import ...` in `scripts/`/`tests/` resolve via `sys.path.insert(0, .../src)` to `src/dqn_agent.py`)* | `src/dqn_agent.py` (pre-4.1 version: flat 5-action `QNetwork`, hard target-copy every 100 steps, no gate/segment hierarchy — verified by full `diff`, ~140 lines differ) | UNUSED, superseded | moved to `archive/legacy/root_scripts/` |
| `test.py` | H | none (not pytest-collected either — its name doesn't match `test_*.py`/`*_test.py`, confirmed via `pytest --collect-only`: 448 tests, same as `pytest tests/`) | — (one-off IGD-vs-population-quality demo plot, no assertions) | UNUSED scratch script | moved to `archive/legacy/root_scripts/` |
| `cec18_main.py` | H | none | — (earliest skeleton: evaluates DF1 in a loop with `# Sau nay them vao day` TODO comments, no NSGA-II/response logic) | UNUSED, superseded by `scripts/train.py` + `src/dynamic_runner.py` | moved to `archive/legacy/root_scripts/` |
| `data/` (`cec15/`, `cec18/`, `requirements.txt`, `README.md`) | F | none in `src/`/`scripts/`/`tests/` (production imports the installed `pymoo` package, not this vendor copy) | — | Not code-referenced, but **intentionally vendored reference material** (own `README.md` explains: offline copy of pymoo's CEC15/18 source + example scripts, "keep vendor for reference only, use `pip install pymoo` for real code") | **KEPT in place** (default per instruction; not archived — its unused-by-code status is the documented, intended state, not a dead-code accident) |
| `paper/` | G | — | — | active (source PDFs for the paper rewrite; new docs go to `docs/`) | kept |
| `results/` | E | — | — | frozen scientific datasets + generated analysis outputs (gitignored) | kept, **never modify** |
| `.claude/`, `.pytest_cache/`, `.gitignore` | — | — | — | tooling/config | kept |

## `src/` (production core)

| path | category | status | notes |
|---|---|---|---|
| `dynamic_runner.py` | A | active | `run_sa_drl` — the full SA-DRL episode loop; the ablation hooks (`mask_change_state`, `disable_memory`) live here (5.3B) |
| `sa_drl_dmoea.py` | A | active | state/action/reward primitives: `build_state`, `compute_dispersion`, `compute_hv_drop`, `compute_reward`, `apply_hierarchical_response`, action functions |
| `dqn_agent.py` | A | active | `HierarchicalQNetwork`, `DQNAgent`, `ReplayBuffer`, Double-DQN target computation |
| `change_detector.py` | A | active | `ChangeDetector` — temporal directional-sensitivity segment-change signal |
| `memory_archive.py` | A | active | `MemoryArchive`, `compute_signature`, `calibrate_scale` |
| `nsga2_pymoo.py` | A | active | **canonical** NSGA-II (pymoo SBX/PM/RankAndCrowding) — used by both `dynamic_runner.py` and `baseline_runner.py` |
| `baseline_runner.py` | A | active | `run_nsga2_baseline` — clean dynamic NSGA-II baseline (4.9), structurally separate from the SA-DRL controller |
| `df_stable.py` | A | active | `DF5Stable`, `DF12Stable`, `DF13Stable` — numerically-hardened DF time formulas (4.10C.1) |
| `experiment_logger.py` | A | active | `ExperimentLogger` — JSONL run logger used by `scripts/train.py` |
| `baselines.py` | A (deprecated shim, kept) | active — **do not archive** | `NSGA2Baseline` class, docstring says "DEPRECATED compatibility shim (4.9)"; still imported by 10 test files as a duck-typed no-op agent. Archiving would break those tests; out of scope to also rewrite them here. |
| `policy_diagnostic.py` | C | active | pure helpers for 5.3A mechanism diagnostics (`decode_gate`, `intervention_rate`, `normalized_entropy`, ...) |
| `stats_analysis.py` | C | active | pure statistics primitives (`holm_correction`, `rank_biserial`, `wilcoxon_paired`, `classify_primary`) — reused unmodified by 5.2/5.3C/5.5A |
| `nsga2.py` | **H** | **UNUSED, superseded** | hand-rolled pre-pymoo NSGA-II; imports `operators`/`sorting`; **zero references** anywhere in `scripts/`/`tests/` (confirmed by grep) — superseded by `nsga2_pymoo.py`. Moved to `archive/legacy/old_nsga2_prepymoo/` |
| `operators.py` | **H** | **UNUSED, superseded** | SBX/polynomial-mutation used only by `src/nsga2.py` (moved with it) |
| `sorting.py` | **H** | **UNUSED, superseded** | dominance/crowding used only by `src/nsga2.py` (moved with it) |

## `scripts/` (experiment entry points + analysis)

| path | category | status | notes |
|---|---|---|---|
| `train.py` | A | active | core training/eval CLI: `mode_frozen` (PRIMARY protocol), `train_on`, `one_run`, `new_agent`, `make_segments`, `problem_setup` — imported by nearly every other script |
| `final_benchmark.py` | B | active | 5.1 — produces the frozen `results/final_benchmark_runs.jsonl` (840 rows). **Do not re-run.** |
| `pilot_3seed.py` | B | completed, kept | pre-5.1 protocol/infrastructure smoke pilot (3 seeds × 5 DFs); its output is referenced in `results/pilot_run.log`/`pilot_summary.json`. Not re-run, but not dead code either — historically informative, kept in place |
| `ablation_5_3B.py` | B | active | 5.3B — produces the frozen `results/ablation_5_3B_runs.jsonl` (2100 rows). **Do not re-run.** |
| `ablation_5_3C_stats.py` | C | active | 5.3C confirmatory statistics on the 5.3B dataset |
| `analyze_benchmark.py` | C | active | 5.2 statistical analysis (produces `results/statistics_*.csv`, `statistical_report.md`) |
| `analyze_policy_diagnostic.py` | C | active | 5.3A mechanism diagnostic analysis |
| `replay_policy_diagnostic.py` | C | active | 5.3A deterministic per-change replay (verified byte-identical against frozen summary metrics) |
| `audit_probe_representativeness.py` | C | completed, kept | 4.10D read-only probe-drift diagnostic |
| `horizon_5_5A.py` | C | active | 5.5A training-horizon 2×2 diagnostic |
| `horizon_5_5A_stats.py` | C | active | 5.5A statistics |
| `jsonl_to_csv.py` | utility | active | small CLI, has its own test (`tests/test_jsonl_to_csv.py`) |

No file in `scripts/` was found unused.

## `tests/` (36 files)

All 36 test files map to a specific implementation phase and are part
of the live `pytest tests/` regression suite (448 passed at time of
audit). None identified as dead. `conftest.py` provides shared pytest
config only.

## Summary of moves

| Moved | To |
|---|---|
| `dqn_agent.py`, `test.py`, `cec18_main.py` | `archive/legacy/root_scripts/` |
| `src/nsga2.py`, `src/operators.py`, `src/sorting.py` | `archive/legacy/old_nsga2_prepymoo/` |

Nothing else was moved. `data/` was audited and explicitly **kept in
place** per instruction (unused-by-code is its documented, intentional
state, not a dead-code accident).
