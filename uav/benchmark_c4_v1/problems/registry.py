"""Problem registry: maps problem names to classes with default configs."""

from __future__ import annotations
from typing import Dict, Type

from .base import DynamicBenchmarkProblem
from .cec2018 import DF1, DF2, DF3, DF4, DF5, DF6, DF7, DF8, DF9, DF10, DF11, DF12, DF13, DF14
from .fda import FDA1, FDA2, FDA3, FDA4, FDA5
from .dmop import dMOP1, dMOP2, dMOP3

PROBLEM_REGISTRY: Dict[str, Type[DynamicBenchmarkProblem]] = {
    "DF1": DF1, "DF2": DF2, "DF3": DF3, "DF4": DF4, "DF5": DF5,
    "DF6": DF6, "DF7": DF7, "DF8": DF8, "DF9": DF9,
    "DF10": DF10, "DF11": DF11, "DF12": DF12, "DF13": DF13, "DF14": DF14,
    "FDA1": FDA1, "FDA2": FDA2, "FDA3": FDA3, "FDA4": FDA4, "FDA5": FDA5,
    "dMOP1": dMOP1, "dMOP2": dMOP2, "dMOP3": dMOP3,
}

BLOCK_A = ["DF1", "DF2", "DF3", "DF4", "DF5", "DF6", "DF7", "DF8", "DF9",
           "DF10", "DF11", "DF12", "DF13", "DF14"]
BLOCK_B = ["FDA1", "FDA2", "FDA3", "FDA4", "FDA5", "dMOP1", "dMOP2", "dMOP3"]
ALL_PROBLEMS = BLOCK_A + BLOCK_B

SMOKE_PROBLEMS = ["DF1", "DF5", "DF7", "DF10", "FDA1", "dMOP2", "dMOP3"]


def get_problem(name: str, n_var: int = 10, nt: int = 10, taut: int = 10) -> DynamicBenchmarkProblem:
    if name not in PROBLEM_REGISTRY:
        raise ValueError(f"Unknown problem: {name}. Available: {list(PROBLEM_REGISTRY.keys())}")
    return PROBLEM_REGISTRY[name](n_var=n_var, nt=nt, taut=taut)
