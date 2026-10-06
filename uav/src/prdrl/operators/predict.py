import numpy as np

from .local import local


def predict(
    X_prev,
    delta_hat,
    n_offspring,
    xl,
    xu,
    sigma_pred,
    rng,
    sigma_local=None,
):
    """
    PREDICT operator - theo PDF hiện tại.

    delta_hat đã được tính bên ngoài từ:

        delta_hat
        = lambda_pred * (mu_{t-1} - mu_{t-2})

    Sau đó sinh offspring:

        z_child
        = z_parent + delta_hat + sigma_pred * xi

        xi ~ N(0, I)

    rồi clip z_child về [0, 1] và chuyển lại decision space.

    Nếu delta_hat is None:
        -> không đủ history
        -> fallback sang LOCAL theo PDF.

    Parameters
    ----------
    X_prev : array, shape (n, D)
        Các nghiệm hiện tại của region, dùng làm parent.

    delta_hat : array shape (D,) hoặc None
        Predicted regional shift.
        None nghĩa là không đủ history.

    n_offspring : int
        Số offspring cần sinh.

    xl, xu : array shape (D,)
        Lower/upper decision bounds.

    sigma_pred : float
        Noise của PREDICT.

    rng : np.random.Generator
        Random generator.

    sigma_local : float or None
        Noise của LOCAL khi phải fallback.
    """

    # --------------------------------------------------
    # 1. Chuẩn hóa input
    # --------------------------------------------------

    X_prev = np.asarray(X_prev, dtype=float)
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    if X_prev.ndim != 2:
        raise ValueError("X_prev must be a 2D array.")

    n_parent, n_var = X_prev.shape

    if n_parent == 0:
        raise ValueError(
            "predict(): X_prev is empty; no parent is available."
        )

    if xl.shape != (n_var,):
        raise ValueError("xl must have shape (D,).")

    if xu.shape != (n_var,):
        raise ValueError("xu must have shape (D,).")

    span = xu - xl

    if np.any(span <= 0):
        raise ValueError(
            "Each upper bound must be greater than lower bound."
        )

    if n_offspring < 0:
        raise ValueError("n_offspring must be non-negative.")

    # Không cần sinh gì
    if n_offspring == 0:
        return np.empty((0, n_var), dtype=float)

    # --------------------------------------------------
    # 2. Thiếu history -> fallback LOCAL
    # --------------------------------------------------

    if delta_hat is None:

        if sigma_local is None:
            raise ValueError(
                "delta_hat is None, so LOCAL fallback requires sigma_local."
            )

        return local(
            X_prev,
            n_offspring,
            xl,
            xu,
            sigma_local,
            rng,
        )

    # --------------------------------------------------
    # 3. Kiểm tra predicted shift
    # --------------------------------------------------

    delta_hat = np.asarray(delta_hat, dtype=float)

    if delta_hat.shape != (n_var,):
        raise ValueError(
            "delta_hat must have shape (D,)."
        )

    if not np.all(np.isfinite(delta_hat)):
        raise ValueError(
            "delta_hat must contain only finite values."
        )

    # --------------------------------------------------
    # 4. Sinh PREDICT offspring
    # --------------------------------------------------

    offspring = []

    for _ in range(n_offspring):

        # Chọn một parent ngẫu nhiên trong region
        idx = rng.integers(n_parent)
        parent = X_prev[idx]

        # Normalize parent về [0, 1]^D
        z_parent = (parent - xl) / span

        # Gaussian noise
        xi = rng.normal(
            loc=0.0,
            scale=1.0,
            size=n_var,
        )

        # Eq. PREDICT:
        #
        # z' = z_parent + delta_hat + sigma_P * xi
        z_child = (
            z_parent
            + delta_hat
            + sigma_pred * xi
        )

        # Repair bounds trong normalized space
        z_child = np.clip(
            z_child,
            0.0,
            1.0,
        )

        # Đưa về decision space thật
        child = xl + z_child * span

        offspring.append(child)

    return np.asarray(
        offspring,
        dtype=float,
    )