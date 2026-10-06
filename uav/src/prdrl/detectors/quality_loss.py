
import numpy as np
def quality_loss(F_minus, F_plus, region_ids, V, theta):
    
    alpha = 1e-12

    F_minus = np.asarray(F_minus, dtype=float)
    F_plus = np.asarray(F_plus, dtype=float)
    region_ids = np.asarray(region_ids, dtype=int)
    V = np.asarray(V, dtype=float)

    R = len(V)
    
    losses = []
    
    for r in range(R):
        mask = region_ids == r
        
        if not np.any(mask):
            losses.append(np.nan)
            continue
        
        F_minus_r = F_minus[mask]
        F_plus_r = F_plus[mask]
        v = V[r]
        
        v_norm = np.linalg.norm(v)

        if v_norm == 0:
            raise ValueError(
                f"Reference direction V[{r}] có norm = 0"
            )

        v = v / v_norm

        region_point_losses = []

        # --------------------------------
        # 3. tính loss cho từng nghiệm
        # --------------------------------
        for f_minus, f_plus in zip(F_minus_r, F_plus_r):

            # ===== TRƯỚC CHANGE =====

            # d1 = projection theo reference direction
            d1_minus = np.dot(f_minus, v)

            # projection point
            proj_minus = d1_minus * v

            # d2 = khoảng cách vuông góc tới direction
            d2_minus = np.linalg.norm(
                f_minus - proj_minus
            )

            # PBI score
            g_minus = (
                d1_minus
                + theta * d2_minus
            )

            # ===== SAU CHANGE =====

            d1_plus = np.dot(f_plus, v)

            proj_plus = d1_plus * v

            d2_plus = np.linalg.norm(
                f_plus - proj_plus
            )

            g_plus = (
                d1_plus
                + theta * d2_plus
            )

            # --------------------------------
            # 4. quality loss
            # --------------------------------
            loss = max(
                0.0,
                g_plus - g_minus
            )

            region_point_losses.append(loss)

        # --------------------------------
        # 5. median loss của region
        # --------------------------------
        loss_r = np.median(
            region_point_losses
        )

        losses.append(loss_r)

    return np.asarray(losses)
        
        


        
        