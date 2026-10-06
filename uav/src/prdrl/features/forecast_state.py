import numpy as np

from prdrl.history.region_history import RegionHistory
def compute_current_predict_motion(mu_t2, mu_t1, lambda_pred=1.0):
    
    mu_t2 = np.asarray(mu_t2, dtype=float)
    mu_t1 = np.asarray(mu_t1, dtype=float)
    
    d_prev = mu_t1 - mu_t2
    delta_hat = lambda_pred * d_prev
    
    return {
        "d_prev": d_prev,
        "delta_hat": delta_hat
    }
    
#upd t-2, t-1, t -> t + 1
#RESERVED FOR FUTURE / NOT USED IN CURRENT PDF PROTOCOL.
def compute_extended_motion(mu_t2, mu_t1, mu_t):
    
    mu_t2 = np.asarray(mu_t2, dtype=float)
    mu_t1 = np.asarray(mu_t1, dtype=float)
    mu_t = np.asarray(mu_t, dtype=float)
    
    d_prev = mu_t1 - mu_t2
    d_curr = mu_t - mu_t1
    
    speed_prev = np.linalg.norm(d_prev)
    speed_curr = np.linalg.norm(d_curr)
    
    if speed_prev > 1e-12 and speed_curr > 1e-12:
        cosine = np.dot(d_prev, d_curr) / (speed_prev * speed_curr)
    else:
        cosine = np.nan
    
    acceleration = d_curr - d_prev

    return {
        "d_prev": d_prev,
        "d_curr": d_curr,
        "speed_prev": speed_prev,
        "speed_curr": speed_curr,
        "cosine": cosine,
        "acceleration": acceleration,
    }
    
def get_predict_motion_from_history(
    history,
    region_id,
    t,
    lambda_pred=1.0,
):
    seq = history.get_centroid_sequence(
        region_id=region_id,
        end_t=t - 1,
        length=2,
    )

    if seq is None:
        return None

    mu_t2 = seq[0]
    mu_t1 = seq[1]

    return compute_current_predict_motion(
        mu_t2,
        mu_t1,
        lambda_pred=lambda_pred,
    )
    
