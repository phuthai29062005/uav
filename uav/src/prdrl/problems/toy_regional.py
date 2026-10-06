import numpy as np


def make_toy_problem(N=100, n_obj=2, R_shift=0, shift_magnitude=0.5, seed=42):
    """
    Toy bi-objective problem: Pareto front đã biết, chỉ MỘT vùng bị dịch.

    Parameters
    ----------
    N              : số nghiệm trên front
    n_obj          : số mục tiêu (chỉ hỗ trợ 2)
    R_shift        : chỉ số vùng sẽ bị dịch (ground truth)
    shift_magnitude: độ dịch trong objective space
    seed           : random seed

    Returns
    -------
    F_minus    : (N, n_obj) objectives trước change
    F_plus     : (N, n_obj) objectives sau change
    X          : (N, D) decision vectors giả (không quan trọng cho detection test)
    pf_before  : true PF trước change (= F_minus vì nghiệm nằm trên front)
    pf_after   : true PF sau change
    R_shift    : vùng bị dịch (ground truth để kiểm tra detector)
    """
    rng = np.random.RandomState(seed)

    # Pareto front đơn giản: f1 = t, f2 = 1 - t, t đều trong [0, 1]
    t = np.linspace(0, 1, N)
    F_minus = np.column_stack([t, 1 - t])

    # Decision vectors giả — chỉ cần tồn tại, không ảnh hưởng detection
    D = 10
    X = rng.rand(N, D)

    # Partition bằng reference directions để biết nghiệm nào thuộc vùng nào
    # (FIX: reference/partition/normalize nằm ở prdrl.core, không phải
    # prdrl.problems -- import tương đối ".xxx" bị sai package, phải là
    # "..core.xxx")
    from ..core.reference import reference
    from ..core.partition import partition
    from ..core.normalize import normalize

    # Chuẩn hoá trước change (F_plus chưa có, tạm dùng F_minus cho cả hai
    # để lấy region assignment — sau đó mới tạo F_plus thật)
    F_temp_minus, _ = normalize(F_minus, F_minus)
    V = reference(n_obj, n_partitions=N // 10)
    region_ids = partition(np.array(F_temp_minus), V)
    R = len(V)

    # Dịch chỉ vùng R_shift
    F_plus = F_minus.copy()
    for i in range(N):
        if region_ids[i] == R_shift:
            F_plus[i] += shift_magnitude

    pf_before = F_minus.copy()
    pf_after = F_plus.copy()

    return F_minus, F_plus, X, pf_before, pf_after, R_shift, region_ids, V, R