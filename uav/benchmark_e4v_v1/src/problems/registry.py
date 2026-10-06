"""Problem registry for CEC2018 DF1-DF14."""

from __future__ import annotations

from typing import Dict, List, Type

from .base import DynamicBenchmarkProblem
from .cec2018 import (
    DF1, DF2, DF3, DF4, DF5, DF6, DF7,
    DF8, DF9, DF10, DF11, DF12, DF13, DF14,
)

BI_OBJECTIVE: List[str] = ["DF1", "DF2", "DF3", "DF4", "DF5", "DF6", "DF7", "DF8", "DF9"]
TRI_OBJECTIVE: List[str] = ["DF10", "DF11", "DF12", "DF13", "DF14"]

_REGISTRY: Dict[str, Type[DynamicBenchmarkProblem]] = {
    "DF1": DF1, "DF2": DF2, "DF3": DF3, "DF4": DF4, "DF5": DF5,
    "DF6": DF6, "DF7": DF7, "DF8": DF8, "DF9": DF9, "DF10": DF10,
    "DF11": DF11, "DF12": DF12, "DF13": DF13, "DF14": DF14,
}


def get_problem(name: str, n_var: int = 10, nt: int = 10, taut: int = 10) -> DynamicBenchmarkProblem:
    cls = _REGISTRY[name.upper()]
    return cls(n_var=n_var, nt=nt, taut=taut)


def get_all_problems(n_var: int = 10, nt: int = 10, taut: int = 10) -> Dict[str, DynamicBenchmarkProblem]:
    return {name: get_problem(name, n_var, nt, taut) for name in _REGISTRY}
