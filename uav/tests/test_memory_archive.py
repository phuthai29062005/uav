"""M1-M7: objective-space memory archive + retrieve-before-store."""
import os
import sys

import numpy as np
import pytest
from pymoo.problems.dynamic.df import DF1

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import dynamic_runner  # noqa: E402
from memory_archive import MemoryArchive, compute_signature  # noqa: E402
from sa_drl_dmoea import compute_env_key  # noqa: E402

D = 10
REF = np.array([2.0, 2.0])
L = 10  # 5 thong ke * 2 objective


def probe(n=10, seed=0):
    return np.random.RandomState(seed).rand(n, D)


def converged_probe(n=10, gens=40, seed=0):
    """
    Probe set GIONG cai runner dung: population da hoi tu sau warm-up.
    Probe uniform-random nam xa PF nen F vuot ref_point o MOI moi
    truong, clip bao hoa va xoa sach khac biet giua cac environment
    — do la artifact cua test, khong phai cua signature.
    """
    from nsga2_pymoo import nsga2_one_generation, seed_nsga2
    seed_nsga2(seed)
    prob = DF1(time=0.0, n_var=D)
    X = np.random.RandomState(seed).rand(n, D)
    F = prob.evaluate(X)
    for _ in range(gens):
        X, F = nsga2_one_generation(X, F, prob, n)
    return X


def sig_at(t, X):
    return compute_signature(X, DF1(time=t, n_var=D), REF)


# --------------------------------------------------- M1: NO SELF-RETRIEVE
def test_never_retrieves_the_environment_being_queried(monkeypatch):
    """
    Tai lan doi thu k (0-based idx), archive chi duoc chua k environment
    TRUOC do — entry cua chinh t chua ton tai nen khong the tu match.
    """
    seen = []
    orig = MemoryArchive.query

    def spy(self, signature):
        seen.append((len(self), np.array(signature, copy=True),
                     [s.copy() for s in self.signatures]))
        return orig(self, signature)

    monkeypatch.setattr(dynamic_runner.MemoryArchive, "query", spy)

    from baselines import NSGA2Baseline
    dynamic_runner.run_sa_drl(
        DF1, n_t=10, tau_t=10, N=20, D=D, n_changes=5, warm_up=10,
        agent=NSGA2Baseline(n_segments=2), segments=[[0], list(range(1, D))],
        seed=0, training=False, ref_point=tuple(REF), n_elite=5)

    assert len(seen) == 5
    for k, (size, q, stored) in enumerate(seen):
        assert size == k, f"lan doi {k}: archive co {size} entry, phai co {k}"
        for e in stored:
            assert np.linalg.norm(q - e) > 1e-9, (
                f"lan doi {k}: archive chua signature trung voi query")


# ------------------------------------------------- M2: PERIODIC REVISIT
@pytest.mark.slow
def test_periodic_environment_is_recognised():
    """
    DF1 tuan hoan theo G(t). Dung dung lifecycle cua runner: tai lan
    doi k, archive chi chua t_0..t_{k-1}. Tai t=4.5 archive da co tron
    mot chu ky nen phai khop t=0.5 (cung pha), KHONG phai t=4.4 (gan
    nhat theo thoi gian).
    """
    X = converged_probe()
    archive = MemoryArchive(max_size=100)
    ts = [round(0.1 * k, 1) for k in range(46)]      # 0.0 .. 4.5
    d_mem, hits = {}, {}
    for k, t in enumerate(ts):
        sig = sig_at(t, X)
        if len(archive):
            res = archive.query(sig)          # RETRIEVE truoc khi STORE
            d_mem[k], hits[k] = res["d_mem"], res["index"]
        archive.store(sig, np.zeros((5, D)))

    # DF1 phu thuoc t CHI qua v = sin(0.5*pi*t), nen v(0.5) == v(1.5)
    # == v(4.5): t=0.5 va t=1.5 la hai moi truong GIONG HET nhau va hoa
    # nhau tuyet doi. Doi hoi dung t=0.5 la over-specified — dieu can
    # chung minh la khop CUNG PHA chu khong phai gan nhat theo thoi gian.
    v = lambda tt: np.sin(0.5 * np.pi * tt)      # noqa: E731
    t_hit = ts[hits[45]]
    print(f"\n  t=4.5 khop t={t_hit} (v={v(t_hit):.6f}, "
          f"v(4.5)={v(4.5):.6f}); gan nhat theo thoi gian la t=4.4 "
          f"(v={v(4.4):.6f})")
    assert abs(v(t_hit) - v(4.5)) < 1e-9, f"khop t={t_hit}, khac pha"
    assert abs(t_hit - 4.4) > 1e-9, "khop entry gan nhat theo THOI GIAN"

    # so voi chu ky DAU (archive chua du mot chu ky nen chua the khop)
    early = np.mean([d_mem[k] for k in range(1, 6)])
    print(f"\n  d_mem tai t=4.5 (da qua tron 1 chu ky) = {d_mem[45]:.6f}"
          f"\n  d_mem trung binh chu ky dau (k=1..5)    = {early:.6f}")
    assert d_mem[45] < early, (d_mem[45], early)


# --------------------------- M3: SIGNATURE DISTINGUISHES ENVIRONMENTS
def test_signature_captures_periodicity():
    """Test quan trong nhat: signature bat duoc tinh tuan hoan."""
    X = converged_probe()
    e = {t: sig_at(t, X) for t in [0.0, 0.5, 1.0, 4.0, 4.5]}
    d = lambda a, b: float(np.linalg.norm(e[a] - e[b]))  # noqa: E731

    print(f"\n  ||e(0.0)-e(4.0)|| = {d(0.0, 4.0):.6f}"
          f"\n  ||e(0.5)-e(4.5)|| = {d(0.5, 4.5):.6f}"
          f"\n  ||e(0.0)-e(0.5)|| = {d(0.0, 0.5):.6f}"
          f"   (nguong: 0.05*sqrt(L)={0.05*np.sqrt(L):.4f}, "
          f"0.10*sqrt(L)={0.10*np.sqrt(L):.4f})")

    assert d(0.0, 4.0) < 0.05 * np.sqrt(L)
    assert d(0.5, 4.5) < 0.05 * np.sqrt(L)
    assert d(0.0, 0.5) > 0.10 * np.sqrt(L)


# ------------------- M4: CENTROID THAT BAI, SIGNATURE THANH CONG
def test_centroid_key_is_blind_to_environment():
    """
    Hai population khac han nhau nhung cung centroid -> compute_env_key
    cu cho key gan nhu trung. Nang hon: env_key khong nhan problem nen
    HAI MOI TRUONG KHAC NHAU cho ra key GIONG HET.
    Signature thi tach duoc.
    """
    rng = np.random.RandomState(1)
    base = rng.rand(20, D)
    pop_tight = np.full((20, D), 0.5) + 0.01 * (base - base.mean(axis=0))
    pop_loose = np.full((20, D), 0.5) + 0.40 * (base - base.mean(axis=0))
    pop_loose = np.clip(pop_loose, 0, 1)

    k1, k2 = compute_env_key(pop_tight), compute_env_key(pop_loose)
    assert np.linalg.norm(k1 - k2) < 1e-6, "hai pop nay phai cung centroid"

    # cung population, HAI moi truong khac nhau -> key trung khop hoan toan
    assert np.array_equal(compute_env_key(pop_tight),
                          compute_env_key(pop_tight))

    s1 = compute_signature(pop_tight, DF1(time=0.0, n_var=D), REF)
    s2 = compute_signature(pop_tight, DF1(time=0.5, n_var=D), REF)
    print(f"\n  ||k1-k2||={np.linalg.norm(k1-k2):.2e}   "
          f"||s(t=0)-s(t=0.5)||={np.linalg.norm(s1-s2):.4f}")
    assert np.linalg.norm(s1 - s2) > 0.10 * np.sqrt(L)


# ------------------------------------------------------ M5: EMPTY ARCHIVE
def test_empty_archive():
    res = MemoryArchive().query(np.zeros(L))
    assert res["has_memory"] is False
    assert res["d_mem"] == 1.0
    assert res["pop"] is None
    assert res["index"] is None


# ----------------------------------------------------- M6: FIFO EVICTION
def test_fifo_eviction():
    a = MemoryArchive(max_size=50)
    first = np.full(L, 0.0)
    for i in range(55):
        a.store(np.full(L, i / 100.0), np.full((3, D), i))
    assert len(a) == 50
    for e in a.signatures:
        assert np.linalg.norm(e - first) > 1e-9, "entry dau tien chua bi day ra"
    assert np.allclose(a.signatures[0], np.full(L, 5 / 100.0))


# -------------------------------------------------------- M7: NO ALIASING
def test_no_aliasing():
    a = MemoryArchive()
    sig, pop = np.zeros(L), np.zeros((4, D))
    a.store(sig, pop)

    pop[0, 0] = 999.0
    res = a.query(sig)
    assert res["pop"][0, 0] != 999.0, "archive phai giu ban copy khi store"

    res["pop"][0, 0] = -1.0
    assert a.query(sig)["pop"][0, 0] != -1.0, "query phai tra ban copy"
