"""Benchmark state construction: maps DMOP problem state to the 9-dim input vector
expected by the frozen C4 controller architecture.

State vector mapping (same 9 dims as frozen core):
  0: severity_r          - normalized change magnitude in region
  1: convergence_loss_r  - IGD of region's solutions to current true PF
  2: occupancy_r         - fraction of population in this region
  3: delta_severity_r    - change in severity since last step
  4: active_region_fraction - fraction of regions that are active (always 1.0 for 2-region)
  5: mean_severity       - mean severity across all regions
  6: max_severity        - max severity across all regions
  7: mean_convergence_loss - mean convergence loss across all regions
  8: max_convergence_loss  - max convergence loss across all regions

Severity is estimated from the objective-space shift between consecutive time steps.
Convergence loss is IGD of the region sub-population to the true PF.
NOTE: true PF is used ONLY for state construction (convergence_loss), never for reward.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


class BenchmarkStateBuilder:
    """Builds 9-dim state vectors for benchmark problems."""

    def __init__(self, n_regions: int = 2):
        self.n_regions = n_regions
        self._prev_centroids: Optional[Dict[int, np.ndarray]] = None
        self._prev_severity: Dict[int, float] = {}

    def reset(self):
        self._prev_centroids = None
        self._prev_severity = {}

    def build_states(self, regions: Dict[int, np.ndarray], X: np.ndarray,
                     F: np.ndarray, true_pf: np.ndarray,
                     pop_size: int) -> Dict[int, np.ndarray]:
        """Build 9-dim state for each region.

        regions: {region_id: indices}, X: (N, D), F: (N, M),
        true_pf: (R, M), pop_size: total pop size.
        Returns: {region_id: state_9} for active regions.
        """
        from ..metrics.igd import igd as compute_igd

        states = {}
        severities = {}
        conv_losses = {}

        for r_id in range(self.n_regions):
            idx = regions.get(r_id, np.array([], dtype=int))
            if len(idx) == 0:
                severities[r_id] = 0.0
                conv_losses[r_id] = 0.0
                continue

            F_r = F[idx]
            centroid = np.mean(F_r, axis=0)

            if self._prev_centroids is not None and r_id in self._prev_centroids:
                shift = np.linalg.norm(centroid - self._prev_centroids[r_id])
                severity = min(1.0, shift)
            else:
                severity = 0.0

            severities[r_id] = severity
            conv_losses[r_id] = compute_igd(F_r, true_pf)

        current_centroids = {}
        for r_id in range(self.n_regions):
            idx = regions.get(r_id, np.array([], dtype=int))
            if len(idx) > 0:
                current_centroids[r_id] = np.mean(F[idx], axis=0)

        sev_values = [severities.get(r, 0.0) for r in range(self.n_regions)]
        cl_values = [conv_losses.get(r, 0.0) for r in range(self.n_regions)]
        mean_sev = np.mean(sev_values) if sev_values else 0.0
        max_sev = np.max(sev_values) if sev_values else 0.0
        mean_cl = np.mean(cl_values) if cl_values else 0.0
        max_cl = np.max(cl_values) if cl_values else 0.0
        n_active = sum(1 for r in range(self.n_regions) if len(regions.get(r, [])) > 0)
        active_frac = n_active / max(1, self.n_regions)

        for r_id in range(self.n_regions):
            idx = regions.get(r_id, np.array([], dtype=int))
            if len(idx) == 0:
                continue

            occupancy = len(idx) / max(1, pop_size)
            sev = severities[r_id]
            prev_sev = self._prev_severity.get(r_id, 0.0)
            delta_sev = sev - prev_sev

            state = np.array([
                sev,                    # 0: severity_r
                conv_losses[r_id],      # 1: convergence_loss_r
                occupancy,              # 2: occupancy_r
                delta_sev,              # 3: delta_severity_r
                active_frac,            # 4: active_region_fraction
                mean_sev,               # 5: mean_severity
                max_sev,                # 6: max_severity
                mean_cl,                # 7: mean_convergence_loss
                max_cl,                 # 8: max_convergence_loss
            ], dtype=np.float32)

            states[r_id] = state

        self._prev_centroids = current_centroids
        self._prev_severity = dict(severities)

        return states


def build_benchmark_state(regions, X, F, true_pf, pop_size, builder):
    """Convenience wrapper."""
    return builder.build_states(regions, X, F, true_pf, pop_size)
