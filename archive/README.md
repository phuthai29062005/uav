# Archive

Files moved here are **not deleted** — kept for history, moved out of
the active tree because they are confirmed unused/superseded (see
`docs/repository_audit.md` for the full audit method). Each entry below
records: original path, why it was archived, what replaced it, and when.

## `legacy/root_scripts/`

| file | original path | reason archived | replacement | task |
|---|---|---|---|---|
| `dqn_agent.py` | `./dqn_agent.py` (repo root) | Pre-4.1 DQN: flat 5-action `QNetwork`, hard target-copy every 100 steps, no gate/segment hierarchy. Confirmed by full `diff` against `src/dqn_agent.py`; zero importers anywhere in `scripts/`/`tests/` (all `from dqn_agent import ...` resolve to `src/dqn_agent.py` via `sys.path.insert`). | `src/dqn_agent.py` (`HierarchicalQNetwork`, gate+segment hierarchy, Polyak soft target) | 6.0 |
| `test.py` | `./test.py` (repo root) | One-off IGD-vs-population-quality demo/plot script, no assertions, not pytest-collected (name doesn't match `test_*.py`), zero references. | `tests/` (36 real pytest files) | 6.0 |
| `cec18_main.py` | `./cec18_main.py` (repo root) | Earliest DF1 skeleton loop with `# them vao day sau` TODO comments; no NSGA-II offspring/response logic; zero references. | `scripts/train.py` + `src/dynamic_runner.py` | 6.0 |

## `legacy/old_nsga2_prepymoo/`

| file | original path | reason archived | replacement | task |
|---|---|---|---|---|
| `nsga2.py` | `src/nsga2.py` | Hand-rolled pre-pymoo NSGA-II implementation. Confirmed zero references in `scripts/`/`tests/` — the production path uses the canonical pymoo-based implementation instead. | `src/nsga2_pymoo.py` | 6.0 |
| `operators.py` | `src/operators.py` | SBX/polynomial-mutation used only by the archived `nsga2.py`. | `pymoo.operators.crossover.sbx.SBX`, `pymoo.operators.mutation.pm.PM` (used directly in `src/nsga2_pymoo.py`) | 6.0 |
| `sorting.py` | `src/sorting.py` | Dominance/crowding-distance used only by the archived `nsga2.py`. | `pymoo.util.nds.non_dominated_sorting.NonDominatedSorting` + `compute_rank_and_crowding()` in `src/nsga2_pymoo.py` | 6.0 |

Full `pytest tests/ -q` was run immediately after each move (448 passed,
no change) to confirm zero behavioral impact.
