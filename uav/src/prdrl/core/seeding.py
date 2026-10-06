import numpy as np


def deterministic_rng(*parts):
    """
    Tao np.random.Generator tat dinh tu mot chuoi so nguyen (Phase 11,
    E1R): vd (experiment_seed, problem_id, change_idx) -- cung bo 3 gia
    tri nay LUON cho cung mot RNG, khac di dai luong nao cung cho RNG
    khac. Dung cho rhi_null() va cho cac thanh phan SDP can "rnd_t" tat
    dinh theo t (vd mu_mix's w, mu_sdeg's dt/pt) de moi lan chay lai
    cung mot experiment seed deu cho ket qua giong het nhau.
    """
    ints = [int(round(p * 1_000_000)) if isinstance(p, float) else int(p)
            for p in parts]
    return np.random.default_rng(np.random.SeedSequence(ints))
