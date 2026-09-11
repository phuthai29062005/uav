import numpy as np


class _NoOpBuffer:
    def push(self, *args, **kwargs):
        pass

    def __len__(self):
        return 0


class NSGA2Baseline:
    """
    DEPRECATED compatibility shim (4.9).

    Live NSGA-II baseline dung run_nsga2_baseline (src/baseline_runner.py)
    va KHONG di qua SA-DRL controller. Class nay giu lai chi de duck-type
    agent trong cac test/khao sat cu; khong dung o baseline path nua.
    """

    def __init__(self, n_segments):
        self.n_segments = n_segments
        self.replay_buffer = _NoOpBuffer()
        self.eps = 0.0  # co thuoc tinh de tuong thich, khong dung

    def select_action(self, state, training=True):
        return 0, np.zeros(self.n_segments, dtype=np.int64)

    def learn(self):
        return None

    def decay_epsilon(self):
        pass
