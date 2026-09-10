import numpy as np


class _NoOpBuffer:
    def push(self, *args, **kwargs):
        pass

    def __len__(self):
        return 0


class NSGA2Baseline:
    """
    Baseline: NSGA-II thuan, khong phan ung khi moi truong doi.
    Dung de doi chung trong Bang 1.

    API giong DQNAgent (duck typing) de run_sa_drl dung chung interface
    ma khong phai sua gi. Luon chon gate=NO_MEMORY va keep (0) cho moi phan doan,
    nen apply_hierarchical_response khong dung vao population -> chi con NSGA-II chay.
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
