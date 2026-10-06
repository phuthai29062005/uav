from .base import DynamicBenchmarkProblem
from .cec2018 import (
    DF1, DF2, DF3, DF4, DF5, DF6, DF7, DF8, DF9,
    DF10, DF11, DF12, DF13, DF14,
)
from .fda import FDA1, FDA2, FDA3, FDA4, FDA5
from .dmop import dMOP1, dMOP2, dMOP3
from .registry import PROBLEM_REGISTRY, get_problem, ALL_PROBLEMS, SMOKE_PROBLEMS, BLOCK_A, BLOCK_B
