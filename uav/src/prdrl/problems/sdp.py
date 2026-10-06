"""
Adapter cho SDP1-SDP15 (Scalable Dynamic multiobjective Problems).

NGUON (Phase 5, E1R): "A Scalable Test Suite for Continuous Dynamic
Multiobjective Optimisation", Shouyong Jiang, Marcus Kaiser, Shengxiang
Yang, Stefanos Kollias, Natalio Krasnogor, arXiv:1903.02510 (Table I,
Section III-B). KHONG co implementation san trong pymoo/pygmo/PlatEMO
(da kiem tra: pymoo 0.6.2 chi co problems/dynamic/df.py va cec2015.py,
KHONG co SDP; PlatEMO/BIMK tren GitHub cung khong co thu muc/file "SDP")
va trang tac gia (homepages.cs.ncl.ac.uk/shouyong.jiang) co chung chi
SSL het han khong the fetch duoc -- nen module nay code truc tiep tu
cong thuc Table I cua paper, trich dan ro rang trong tung ham.

Framework chung (eq. 1-2):
    f_i(x,t) = (1 + g(x,t)) * mu_i(x,t) + nu(t),   i = 1..M
    PF dat duoc khi g(x,t) = 0.

DA IMPLEMENT (13/15, du 7 bai pre-registered cho E1b):
    SDP2, SDP3, SDP4, SDP5, SDP6, SDP7, SDP8, SDP9, SDP10, SDP11,
    SDP13, SDP14, SDP15

CHUA IMPLEMENT (STOP AND REPORT, khong doan cong thuc -- Phase 16
"Do NOT guess formulas"):
    SDP1  -- can PS ngau nhien de quy (eq.14: x_bar_i,t phu thuoc
             x_bar_i,{t-1}), kien truc stateless evaluate_sdp(id, X, t)
             hien tai khong tu nhien bieu dien duoc chuoi de quy nay
             (se can them tham so lich su rieng). KHONG bat buoc cho
             tap pre-registered (SDP2,4,5,8,9,10,11).
    SDP12 -- (a) so bien n THAY DOI theo thoi gian (eq.15, nt in
             [10,20]) trong khi Problem interface (va evaluate_sdp(X,t))
             gia dinh so cot D CO DINH cho 1 lan goi; (b) dinh nghia
             bang I cua SDP12 chi dung M-1 objectives
             ("fi=1:M-1(x,t)=(1+g)(1-xi)µlinear,i") khac voi cau truc
             M-objective chuan cua cac SDP khac, can lam ro them truoc
             khi implement. KHONG bat buoc cho tap pre-registered.

Mot so o trong Table I (dang PDF -> text) bi trich xuat ra 2 case
khong day du cho eq.(3) mu_linear (chi thay case "i=1" va "2<=i<M",
KHONG co case "i=M" ro rang) -- xac minh bang 2 phuong phap trich
xuat PDF doc lap (pypdf mode mac dinh va mode layout) deu cho CUNG mot
ket qua, nen day la loi typeset that trong chinh ban PDF (khong phai
loi trich xuat cua chung ta). Tuy nhien eq.(18) mu_ldeg -- ro rang noi
"dua tren simplex-shape PF cua (3)" -- co CUNG cau truc nhung case
duoc ghi day du va RO RANG ("neu 1<=i<M" / "neu i=M"), xac nhan cau
truc dung cua eq(3) la:
    mu_linear,i = (1-xi) * prod_{j<i} xj          neu 1<=i<M
    mu_linear,M = x1*x2*...*x_{M-1}               neu i=M
(giong het pattern cua mu_sphere (eq.4), chi thay sin/cos bang
(1-x)/x). Dung reconstruction nay (co trich dan/ly do ro rang) cho
SDP7/SDP14, KHONG phai doan mu.
"""
import numpy as np


# =====================================================================
# Seeding tat dinh cho "rnd_t" (Phase 11 tinh than: khong dung
# np.random khong seed o ben trong mot benchmark function -- can tat
# dinh theo (seed, problem_id, t) de MOI lan goi evaluate_sdp o CUNG
# mot t (vd nhieu the he NSGA-II trong cung 1 environment) deu cho
# CUNG mot gia tri rnd_t, nhung KHAC t thi rnd_t khac.
# =====================================================================

def _rnd_t(seed, problem_id, t, n=1):
    ss = np.random.SeedSequence([int(seed), int(problem_id),
                                 int(round(float(t) * 1_000_000))])
    rng = np.random.default_rng(ss)
    return rng.random(n)


def _randint_t(seed, problem_id, t, low, high, n=1):
    """rndi_t(a,b): so nguyen ngau nhien tat dinh trong [low,high]."""
    ss = np.random.SeedSequence([int(seed), int(problem_id),
                                 int(round(float(t) * 1_000_000)), 777])
    rng = np.random.default_rng(ss)
    return rng.integers(low, high + 1, size=n)


# =====================================================================
# Component mu-functions (Section III-B)
# =====================================================================

def _mu_sum_reciprocal(Y, t):
    """eq.6, Y=(N,M-1) vi tri (position) variables x1..x_{M-1}."""
    N, Mm1 = Y.shape
    M = Mm1 + 1
    mu = np.empty((N, M))
    col_sum = Y.sum(axis=1)
    for i in range(Mm1):
        others = col_sum - Y[:, i]
        mu[:, i] = (1.0 + t + others) / Y[:, i]
    mu[:, M - 1] = col_sum / (1.0 + t)
    return mu


def _mu_knee(Y, w):
    """eq.7, Y=(N,M-1)."""
    N, Mm1 = Y.shape
    M = Mm1 + 1
    terms = Y + 0.05 * np.sin(w * np.pi * Y)   # (N, M-1)
    mu = np.empty((N, M))
    cumprod = np.ones(N)
    for i in range(Mm1):
        mu[:, i] = (1.0 - Y[:, i] + 0.05 * np.sin(w * np.pi * Y[:, i])) * cumprod
        cumprod = cumprod * terms[:, i]
    mu[:, M - 1] = cumprod
    return mu


def _mu_mix(Y, w):
    """eq.8, Y=(N,M-1)."""
    N, Mm1 = Y.shape
    M = Mm1 + 1
    mu = np.empty((N, M))
    if Mm1 >= 2:
        mu[:, :Mm1 - 1] = Y[:, :Mm1 - 1]
    s = Y.mean(axis=1)
    mu[:, M - 2] = s + 0.05 * np.sin(w * np.pi * s)
    mu[:, M - 1] = 1.0 - s + 0.05 * np.sin(w * np.pi * s)
    return mu


def _mu_disc1(Y, kt):
    """eq.10, Y=(N,M-1). Ghi chu: ban PDF hien thi mot cau truc tong
    long nhau bat thuong cho case i=M:
        sum_{j=1}^{M-1} [ sin^2(0.5*pi*xj) + sum_{j=1}^{M-1} sin(0.5*pi*xj)*cos^2(kt*pi*xj) ]
    (tong ngoai KHONG phu thuoc chi so trong cua tong trong). Da xac
    minh bang 2 phuong phap trich xuat PDF doc lap, deu cho CUNG ket
    qua nay -- giu nguyen literal nhu paper, KHONG tu "sua" thanh dang
    "gon" hon vi do se la doan cong thuc. Tuong duong:
        Sigma_a = sum_j sin^2(0.5*pi*xj)
        Sigma_b = sum_j sin(0.5*pi*xj)*cos^2(kt*pi*xj)
        mu_M    = Sigma_a + (M-1)*Sigma_b
    """
    N, Mm1 = Y.shape
    M = Mm1 + 1
    mu = np.empty((N, M))
    mu[:, :Mm1] = np.cos(0.5 * np.pi * Y) ** 2
    sigma_a = np.sum(np.sin(0.5 * np.pi * Y) ** 2, axis=1)
    sigma_b = np.sum(np.sin(0.5 * np.pi * Y) * np.cos(kt * np.pi * Y) ** 2, axis=1)
    mu[:, M - 1] = sigma_a + Mm1 * sigma_b
    return mu


def _mu_disc2(Y, rt):
    """eq.11, Y=(N,M-1)."""
    N, Mm1 = Y.shape
    M = Mm1 + 1
    mu = np.empty((N, M))
    mu[:, :Mm1] = Y
    s = np.sum(Y ** 2, axis=1) / Mm1
    with np.errstate(invalid="ignore"):
        base = -np.sin(2.5 * np.pi * s)
        # co so co the am -> **rt (rt la so nguyen tu floor(10|sin|),
        # luon >=0) van hop le voi so am khi rt nguyen; dung np.sign
        # de tranh NaN tu ham power voi co so am, so mu khong nguyen.
        rt_int = int(round(rt))
        powered = np.sign(base) ** rt_int * np.abs(base) ** rt_int if rt_int != 0 else np.ones_like(base)
    mu[:, M - 1] = 2.0 - s - np.sqrt(np.clip(s, 0, None)) * powered
    return mu


def _mu_sphere_from_y(Yang):
    """eq.4, Yang=(N,M-1) da la y=0.5*pi*x (hoac bien the y khac, vd
    eq.20 shrink -- goi ham nay voi Yang tuong ung)."""
    N, Mm1 = Yang.shape
    M = Mm1 + 1
    mu = np.empty((N, M))
    cos_cum = np.ones(N)
    for i in range(Mm1):
        mu[:, i] = np.sin(Yang[:, i]) * cos_cum
        cos_cum = cos_cum * np.cos(Yang[:, i])
    mu[:, M - 1] = cos_cum
    return mu


def _mu_detect(x1, Yang_rest, alpha_t, kt):
    """eq.13 + 'mu_detect,i=mu_sphere,i for i=2..M'. Yang_rest =
    0.5*pi*x cho cac bien vi tri con lai (x2..x_{M-1}), shape
    (N, M-2) (M-2 vi obj-1 dac biet, con lai M-1 obj deu la sphere
    tren M-2 bien vi tri + x1 dong vai tro nhu "y0" trong tich cos).
    De don gian va nhat quan voi cau truc sphere chuan (M-1 bien vi
    tri cho M obj), ta dung TOAN BO (x1, x2..x_{M-1}) lam input cho
    sphere chuan (M-1 cot), roi GHI DE cot dau tien bang cong thuc
    eq.13."""
    y1 = 0.5 * np.pi * x1
    Yang = np.concatenate([y1[:, None], Yang_rest], axis=1)
    mu = _mu_sphere_from_y(Yang)
    special = np.where(
        x1 <= alpha_t,
        np.abs(kt * (np.cos(0.5 * np.pi * x1) - np.cos(0.5 * np.pi * alpha_t))
              + np.sin(0.5 * np.pi * alpha_t)),
        np.sin(0.5 * np.pi * x1),
    )
    mu[:, 0] = special
    return mu


def _mu_dobj(Y):
    """eq.17, Y=(N,Mt) -- CHU Y: dung Mt bien vi tri (khong phai Mt-1)
    theo dung nhu paper viet (khong co case rieng cho i=Mt)."""
    N, M = Y.shape
    mu = np.empty((N, M))
    cos_cum = np.ones(N)
    for i in range(M):
        mu[:, i] = np.sin(Y[:, i]) * cos_cum
        cos_cum = cos_cum * np.cos(Y[:, i])
    return mu


def _mu_linear(Y):
    """eq.3 (reconstructed -- xem docstring module o dau file), Y=(N,M-1)."""
    N, Mm1 = Y.shape
    M = Mm1 + 1
    mu = np.empty((N, M))
    cumprod = np.ones(N)
    for i in range(Mm1):
        mu[:, i] = (1.0 - Y[:, i]) * cumprod
        cumprod = cumprod * Y[:, i]
    mu[:, M - 1] = cumprod
    return mu


def _mu_ldeg(Yld, g):
    """eq.18, Yld=(N,M-1) da la 'y' (khong phai x truc tiep -- xem
    _sdp14_y). Dung CUNG cau truc voi _mu_linear nhung so hang dau la
    (1+g-yi) thay vi (1-yi)."""
    N, Mm1 = Yld.shape
    M = Mm1 + 1
    mu = np.empty((N, M))
    cumprod = np.ones(N)
    g = np.asarray(g, dtype=float)
    for i in range(Mm1):
        mu[:, i] = (1.0 + g - Yld[:, i]) * cumprod
        cumprod = cumprod * Yld[:, i]
    mu[:, M - 1] = cumprod
    return mu


def _mu_sdeg(Ysd):
    """eq.19, Ysd=(N,M-1) -- cung cau truc _mu_sphere_from_y (chi khac
    cach tinh y ben ngoai, xem _sdp15_y)."""
    return _mu_sphere_from_y(Ysd)


def _gnsc(Y, kt):
    """eq.9, Y=(N,M-1)."""
    r = 1.0 - (kt % 2)
    inner = np.floor(kt * (2 * Y - r)) * np.pi / 2.0
    prod = np.prod(np.sin(inner), axis=1)
    return 1.0 + np.abs(prod)


def _gmulti(Xdist, seed, t):
    """eq.12, Xdist=(N,k) cac bien distance-related (x_M..x_n).

    FIX (t-sensitivity regression, phat hien qua smoke-test): ban dau
    dung hk=k co dinh cho MOI k=1..5, hoan toan bo qua phan "global
    optima duoc dat la h_kt=0.5 voi kt=ceil(5*rndt), tai thoi diem t"
    -- day chinh la co che "deceptive change" trung tam cua SDP7 (Fig.
    6 trong paper: local-vs-global minimum doi cho theo t). Thieu no,
    gmulti hoan toan KHONG phu thuoc t (da xac nhan bang smoke-test:
    F(t=0) == F(t=0.3) het cho SDP7). Sua: moi lan goi, xac dinh kt
    tat dinh theo (seed, t), ghi de h[kt-1] = 0.5.
    """
    hk = np.arange(1, 6, dtype=float)
    rnd = _rnd_t(seed, 7, t)[0]
    kt = int(np.ceil(5 * rnd))
    kt = min(max(kt, 1), 5)
    hk[kt - 1] = 0.5
    yk = 2.0 * (hk - 1.0)
    # min_{k=1..5} { hk + 10*(10*xi - yk)^2 }, tinh cho tung xi roi
    # trung binh theo k=M..n (eq.12: (1/(n-M+1)) * sum_i min_k{...}).
    xi = Xdist[:, :, None]                       # (N,k,1)
    val = hk[None, None, :] + 10.0 * (10.0 * xi - yk[None, None, :]) ** 2
    min_k = val.min(axis=2)                       # (N,k)
    return min_k.mean(axis=1)


def _gfavour(Xdist, t):
    """eq.21, Xdist=(N,k)."""
    p = Xdist - np.abs(np.sin(0.5 * np.pi * t))
    inside = -0.9 * p ** 2 + np.abs(p) ** 0.6
    outside = p ** 2
    # Phi(x,t) (subspace) can x day du de kiem tra tong x1..x_{M-1} --
    # xem _sdp11_eval, ham nay CHI tinh phan tong theo Xdist gia dinh
    # da biet o trong/ngoai Phi qua mask truyen vao tu ben ngoai.
    return inside, outside


# =====================================================================
# Common bounds helper
# =====================================================================

def _pos_dist_bounds(n_var, n_obj, pos_lo, pos_hi, dist_lo, dist_hi,
                     n_pos=None):
    n_pos = (n_obj - 1) if n_pos is None else n_pos
    xl = np.empty(n_var)
    xu = np.empty(n_var)
    xl[:n_pos] = pos_lo
    xu[:n_pos] = pos_hi
    xl[n_pos:] = dist_lo
    xu[n_pos:] = dist_hi
    return xl, xu


# =====================================================================
# Per-problem: bounds
# =====================================================================

def get_sdp_bounds(problem_id, n_obj, n_var=10):
    if problem_id == 2:
        return _pos_dist_bounds(n_var, n_obj, 1.0, 4.0, -1.0, 1.0, n_pos=n_obj - 1)
    if problem_id in (3, 4, 8, 10):
        return _pos_dist_bounds(n_var, n_obj, 0.0, 1.0, -1.0, 1.0, n_pos=n_obj - 1)
    if problem_id in (5, 6, 7, 9, 11, 13, 14, 15):
        return np.zeros(n_var), np.ones(n_var)
    raise NotImplementedError(
        f"SDP{problem_id}: khong implement (xem docstring dau file sdp.py "
        f"-- SDP1/SDP12 bi hoan lai co ly do ro rang, khong phai bug).")


def get_sdp_n_obj(problem_id):
    """M khuyen nghi/co dinh theo paper goc (chi mang tinh tham khao --
    evaluate_sdp() LUON yeu cau n_obj tuong minh, KHONG tu suy ra)."""
    if problem_id in (14, 15):
        return 5   # paper co dinh M=5 de mo phong PF degeneration
    return None


# =====================================================================
# Per-problem: evaluate
# =====================================================================

def _split(X, n_pos):
    return X[:, :n_pos], X[:, n_pos:]


def evaluate_sdp(problem_id, X, t, n_obj, seed=0):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X phai shape (N,D), nhan duoc {X.shape}")
    t = float(t)
    n_pos = n_obj - 1

    if problem_id == 2:
        Y, Xd = _split(X, n_pos)
        g = np.sin(np.pi / 8.0 * Y[:, 0]) * np.sum(
            (Xd - np.cos(t + 2 * Y[:, 0])[:, None]) ** 2, axis=1)
        mu = _mu_sum_reciprocal(Y, t)
        F = mu.copy()
        F[:, :n_obj - 1] = (1.0 + g)[:, None] * mu[:, :n_obj - 1]
        return F

    if problem_id == 3:
        Y, Xd = _split(X, n_pos)
        yi = Xd - np.cos(t)
        kt = np.floor(5 * np.abs(np.sin(np.pi * t)))
        g = np.sum(4 * yi ** 2 - np.cos(2 * kt * np.pi * yi) + 1, axis=1)
        mu = _mu_knee(Y, w=kt)
        return (1.0 + g)[:, None] * mu

    if problem_id == 4:
        Y, Xd = _split(X, n_pos)
        x1 = Y[:, 0]
        # g = sum_{i=M}^{n} (xi - cos(t+x1+x_{i-1}))^2 : x_{i-1} cho
        # cot dau tien cua Xd la Y[:,-1] (x_{M-1}, bien vi tri cuoi
        # cung), cac cot sau la CHINH cot distance TRUOC do (linkage
        # tuan tu).
        prev = Y[:, -1] if Y.shape[1] > 0 else x1
        g_terms = np.empty_like(Xd)
        for j in range(Xd.shape[1]):
            target = np.cos(t + x1 + prev)
            g_terms[:, j] = (Xd[:, j] - target) ** 2
            prev = Xd[:, j]
        g = g_terms.sum(axis=1)
        rnd = _rnd_t(seed, 4, t)[0]
        w = np.sign(rnd - 0.5) * np.floor(6 * abs(np.sin(0.5 * np.pi * t)))
        mu = _mu_mix(Y, w=w)
        return (1.0 + g)[:, None] * mu

    if problem_id == 5:
        Gt = abs(np.sin(0.5 * np.pi * t))
        Y, Xd = _split(X, n_pos)
        x1 = Y[:, 0]
        g = Gt + np.sum((Xd - 0.5 * Gt * x1[:, None]) ** 2, axis=1)
        yi = (np.pi / 6.0) * Gt + (np.pi / 2.0 - np.pi / 3.0 * Gt) * Y
        mu_full = _mu_sphere_from_y(yi)
        mu_rev = mu_full[:, ::-1]   # mu_sphere,M-i+1
        return (1.0 + g)[:, None] * mu_rev

    if problem_id == 6:
        Y, Xd = _split(X, n_pos)
        g = np.sum((Xd - 0.5) ** 2 * (1 + np.abs(np.cos(8 * np.pi * Xd))), axis=1)
        alpha_t = 0.5 * abs(np.sin(np.pi * t))
        kt = 10 * np.cos(2.5 * np.pi * t)
        x1 = Y[:, 0]
        Yang_rest = 0.5 * np.pi * Y[:, 1:]
        mu_full = _mu_detect(x1, Yang_rest, alpha_t, kt)
        mu_rev = mu_full[:, ::-1]   # mu_detect,M-i+1
        return (1.0 + g)[:, None] * mu_rev

    if problem_id == 7:
        Y, Xd = _split(X, n_pos)
        g = _gmulti(Xd, seed, t)
        mu = _mu_linear(Y)
        return (0.5 + g)[:, None] * mu

    if problem_id == 8:
        Y, Xd = _split(X, n_pos)
        x1 = Y[:, 0]
        kt = np.floor(10 * np.sin(0.5 * np.pi * t))
        gnsc = _gnsc(Y, kt)
        g = np.sum((Xd - np.sin(t * x1)[:, None]) ** 2, axis=1) + gnsc
        yang = 0.5 * np.pi * Y
        mu_full = _mu_sphere_from_y(yang)
        mu_rev = mu_full[:, ::-1]
        return (1.0 + g)[:, None] * mu_rev

    if problem_id == 9:
        Y, Xd = _split(X, n_pos)
        target = (1.0 / np.pi) * np.abs(np.arctan(1.0 / np.tan(3 * np.pi * t ** 2)))\
            if abs(np.sin(3 * np.pi * t ** 2)) > 1e-9 else 0.0
        g_dist = np.sum((Xd - target) ** 2, axis=1)
        kt = np.floor(6 * np.sin(0.5 * np.pi * t))
        mu = _mu_disc1(Y, kt)
        Gt = abs(np.sin(0.5 * np.pi * t))
        # (1+g*max(0,i-(M-1)))*mu_i: multiplier la 1 cho i<M (khong co
        # g), va (1+g_dist) chi cho i=M -- xem docstring dau file ve
        # cach doc "max(0,i-M-1)" trong Table I la "max(0,i-(M-1))".
        F = np.empty_like(mu)
        F[:, :-1] = mu[:, :-1] + Gt
        F[:, -1] = (1.0 + g_dist) * mu[:, -1] + Gt
        return F

    if problem_id == 10:
        Y, Xd = _split(X, n_pos)
        x1 = Y[:, 0]
        target = np.sin(x1 + 0.5 * np.pi * t)
        g_dist = np.sum((Xd - target[:, None]) ** 2, axis=1)
        rt = np.floor(10 * abs(np.sin(0.5 * np.pi * t)))
        mu = _mu_disc2(Y, rt)
        F = mu.copy()
        F[:, -1] = (1.0 + g_dist) * mu[:, -1]
        return F

    if problem_id == 11:
        Y, Xd = _split(X, n_pos)
        p = Xd - abs(np.sin(0.5 * np.pi * t))
        s = np.sum(Y, axis=1)
        lo = 3 * t - np.floor(3 * t)
        hi = 3 * t + 0.2 - np.floor(3 * t + 0.2)
        in_phi = (s >= lo) & (s <= hi)
        g = np.where(in_phi[:, None], -0.9 * p ** 2 + np.abs(p) ** 0.6, p ** 2)
        g = g.sum(axis=1)
        yang = 0.5 * np.pi * Y
        mu = _mu_sphere_from_y(yang)
        return (1.0 + g)[:, None] * mu

    if problem_id == 13:
        Mt = n_obj
        Y = X[:, :Mt]
        Xd = X[:, Mt:]
        n = X.shape[1]
        idx = np.arange(Mt + 1, n + 1, dtype=float)   # i = M+1..n (1-indexed)
        # FIX (t-sensitivity regression, phat hien qua smoke-test):
        # "it" trong "target = it/(M+it)" la i*t (tich chi so i voi
        # thoi gian t), KHONG phai chi so i suong -- ban dau thieu *t
        # nen g hoan toan khong phu thuoc t (F(t=0)==F(t=0.3)).
        it = idx * t
        target = it / (Mt + it)
        g = np.sum((Xd - target[None, :]) ** 2, axis=1)
        yk = np.pi * (Y + 1.0) / 6.0
        mu = _mu_dobj(yk)
        return (1.0 + g)[:, None] * mu

    if problem_id == 14:
        Y, Xd = _split(X, n_pos)
        M = n_obj
        dt = 1 + int(np.floor(abs((M - 2) * np.cos(0.5 * np.pi * t))))
        g = np.sum((Xd - 0.5) ** 2, axis=1)
        yld = np.empty_like(Y)
        for i in range(Y.shape[1]):
            if i < dt:
                yld[:, i] = Y[:, i]
            else:
                yld[:, i] = 0.5 + Y[:, i] * (abs(np.sin(0.5 * np.pi * t)) ** g)
        mu = _mu_ldeg(yld, g)
        return (1.0 + g)[:, None] * mu

    if problem_id == 15:
        Y, Xd = _split(X, n_pos)
        M = n_obj
        dt = int(_randint_t(seed, 15, t, 1, M - 1)[0])
        pt = int(_randint_t(seed, 15, t, 1, M - 1)[0])
        g = np.sum((Xd - dt / (M - 1)) ** 2, axis=1)
        ysd = np.empty_like(Y)
        gsafe = np.abs(np.sin(0.5 * np.pi * t)) ** g
        for i in range(Y.shape[1]):
            k = (pt + i) % (M - 1)   # 0-indexed: (pt+(i+1)-1) mod (M-1)
            if k < dt:
                ysd[:, i] = 0.5 * np.pi * Y[:, k]
            else:
                denom = 1.0 + Y[:, k] * gsafe
                arg = np.clip((0.5 * np.sqrt(2.0)) / np.maximum(denom, 1e-12), -1.0, 1.0)
                ysd[:, i] = np.arccos(arg)
        mu = _mu_sdeg(ysd)
        return (1.0 + g)[:, None] * mu

    raise NotImplementedError(
        f"SDP{problem_id}: khong implement (xem docstring dau file sdp.py "
        f"-- SDP1/SDP12 bi hoan lai co ly do ro rang, khong phai bug).")


# =====================================================================
# True PF: sample position vars trong box, dat distance vars = gia tri
# zero-hoa g (moi g deu la tong binh phuong (xi-target)^2 -> zero khi
# xi=target dung), roi goi evaluate_sdp voi g=0.
# =====================================================================

def get_sdp_true_pf(problem_id, t, n_obj, n_points=500, n_var=10, seed=0):
    rng = np.random.default_rng(seed)
    xl, xu = get_sdp_bounds(problem_id, n_obj, n_var=n_var)
    n_pos = n_obj - 1 if problem_id != 13 else n_obj
    X = np.empty((n_points, n_var))
    X[:, :n_pos] = rng.uniform(xl[:n_pos], xu[:n_pos], size=(n_points, n_pos))

    t = float(t)
    if problem_id == 2:
        x1 = X[:, 0]
        X[:, n_pos:] = np.cos(t + 2 * x1)[:, None]
    elif problem_id == 3:
        X[:, n_pos:] = np.cos(t)
    elif problem_id == 4:
        x1 = X[:, 0]
        prev = X[:, n_pos - 1]
        for j in range(n_pos, n_var):
            X[:, j] = np.cos(t + x1 + prev)
            prev = X[:, j]
    elif problem_id == 5:
        Gt = abs(np.sin(0.5 * np.pi * t))
        x1 = X[:, 0]
        X[:, n_pos:] = 0.5 * Gt * x1[:, None]
    elif problem_id == 6:
        X[:, n_pos:] = 0.5
    elif problem_id == 7:
        # gmulti dat GIA TRI NHO NHAT co the (=0.5, khong phai 0 --
        # gmulti = min_k{hk+10(10xi-yk)^2}, va h_kt da duoc ghi de =0.5
        # boi _gmulti) tai xi = yk/10 voi k=kt (chi so cua local minimum
        # duoc lam "global" tai thoi diem t nay). Dung CUNG kt tat dinh
        # nhu _gmulti() de nhat quan.
        rnd = _rnd_t(seed, 7, t)[0]
        kt = min(max(int(np.ceil(5 * rnd)), 1), 5)
        yk_target = 2.0 * (kt - 1.0)
        X[:, n_pos:] = yk_target / 10.0
    elif problem_id == 8:
        x1 = X[:, 0]
        X[:, n_pos:] = np.sin(t * x1)[:, None]
    elif problem_id == 9:
        target = (1.0 / np.pi) * abs(np.arctan(1.0 / np.tan(3 * np.pi * t ** 2))) \
            if abs(np.sin(3 * np.pi * t ** 2)) > 1e-9 else 0.0
        X[:, n_pos:] = target
    elif problem_id == 10:
        x1 = X[:, 0]
        X[:, n_pos:] = np.sin(x1 + 0.5 * np.pi * t)[:, None]
    elif problem_id == 11:
        X[:, n_pos:] = abs(np.sin(0.5 * np.pi * t))
    elif problem_id == 13:
        Mt = n_obj
        n = n_var
        idx = np.arange(Mt + 1, n + 1, dtype=float)
        it = idx * t
        target = it / (Mt + it)
        X[:, n_pos:] = target[None, :]
    elif problem_id == 14:
        X[:, n_pos:] = 0.5
    elif problem_id == 15:
        M = n_obj
        dt = int(_randint_t(seed, 15, t, 1, M - 1)[0])
        X[:, n_pos:] = dt / (M - 1)
    else:
        raise NotImplementedError(f"SDP{problem_id}: khong implement.")

    F = evaluate_sdp(problem_id, X, t, n_obj, seed=seed)
    return F
