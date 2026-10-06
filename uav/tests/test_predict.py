import numpy as np

from prdrl.operators.predict import predict


def test_predict_no_noise_exact_shift():
    rng = np.random.default_rng(123)

    X_prev = np.array([
        [0.2, 0.4],
    ])

    delta_hat = np.array([0.1, 0.05])

    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])

    children = predict(
        X_prev=X_prev,
        delta_hat=delta_hat,
        n_offspring=1,
        xl=xl,
        xu=xu,
        sigma_pred=0.0,
        rng=rng,
        sigma_local=0.02,
    )

    expected = np.array([
        [0.3, 0.45],
    ])

    np.testing.assert_allclose(
        children,
        expected,
        atol=1e-12,
    )


def test_predict_clips_to_bounds():
    rng = np.random.default_rng(123)

    X_prev = np.array([
        [0.95, 0.05],
    ])

    delta_hat = np.array([0.2, -0.2])

    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])

    children = predict(
        X_prev=X_prev,
        delta_hat=delta_hat,
        n_offspring=1,
        xl=xl,
        xu=xu,
        sigma_pred=0.0,
        rng=rng,
        sigma_local=0.02,
    )

    expected = np.array([
        [1.0, 0.0],
    ])

    np.testing.assert_allclose(
        children,
        expected,
        atol=1e-12,
    )


def test_predict_zero_offspring():
    rng = np.random.default_rng(123)

    X_prev = np.array([
        [0.2, 0.4],
        [0.3, 0.5],
    ])

    delta_hat = np.array([0.1, 0.1])

    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])

    children = predict(
        X_prev=X_prev,
        delta_hat=delta_hat,
        n_offspring=0,
        xl=xl,
        xu=xu,
        sigma_pred=0.03,
        rng=rng,
        sigma_local=0.02,
    )

    assert children.shape == (0, 2)


def test_predict_multiple_offspring_shape():
    rng = np.random.default_rng(123)

    X_prev = np.array([
        [0.2, 0.4],
        [0.3, 0.5],
    ])

    delta_hat = np.array([0.1, 0.05])

    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])

    children = predict(
        X_prev=X_prev,
        delta_hat=delta_hat,
        n_offspring=5,
        xl=xl,
        xu=xu,
        sigma_pred=0.03,
        rng=rng,
        sigma_local=0.02,
    )

    assert children.shape == (5, 2)

    assert np.all(children >= 0.0)
    assert np.all(children <= 1.0)


def test_predict_missing_history_falls_back_to_local():
    rng_predict = np.random.default_rng(123)
    rng_local = np.random.default_rng(123)

    X_prev = np.array([
        [0.5, 0.5],
    ])

    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])

    sigma_local = 0.02

    children_predict = predict(
        X_prev=X_prev,
        delta_hat=None,
        n_offspring=3,
        xl=xl,
        xu=xu,
        sigma_pred=0.03,
        rng=rng_predict,
        sigma_local=sigma_local,
    )

    # Tự tính expected LOCAL với cùng random seed
    expected = []

    for _ in range(3):
        idx = rng_local.integers(len(X_prev))
        parent = X_prev[idx]

        z_parent = (parent - xl) / (xu - xl)

        xi = rng_local.normal(
            loc=0.0,
            scale=1.0,
            size=len(parent),
        )

        z_child = z_parent + sigma_local * xi
        z_child = np.clip(z_child, 0.0, 1.0)

        child = xl + z_child * (xu - xl)

        expected.append(child)

    expected = np.asarray(expected)

    np.testing.assert_allclose(
        children_predict,
        expected,
        atol=1e-12,
    )


def test_predict_rejects_empty_parent_region():
    rng = np.random.default_rng(123)

    X_prev = np.empty((0, 2))

    delta_hat = np.array([0.1, 0.1])

    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])

    try:
        predict(
            X_prev=X_prev,
            delta_hat=delta_hat,
            n_offspring=1,
            xl=xl,
            xu=xu,
            sigma_pred=0.03,
            rng=rng,
            sigma_local=0.02,
        )

        assert False, "Expected ValueError for empty X_prev"

    except ValueError:
        pass


def test_predict_rejects_invalid_delta_shape():
    rng = np.random.default_rng(123)

    X_prev = np.array([
        [0.2, 0.4],
    ])

    delta_hat = np.array([0.1])  # sai shape

    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])

    try:
        predict(
            X_prev=X_prev,
            delta_hat=delta_hat,
            n_offspring=1,
            xl=xl,
            xu=xu,
            sigma_pred=0.03,
            rng=rng,
            sigma_local=0.02,
        )

        assert False, "Expected ValueError for invalid delta_hat shape"

    except ValueError:
        pass