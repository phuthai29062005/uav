"""
Toy regional-shift problem voi ground truth (dac ta muc 10.1, Experiment
E0 - sanity: "Tao toy problems trong do chi mot trade-off region bi dich.
Ky vong RHI tang va detector chi ra dung region.").

Thiet ke: lam viec TRUC TIEP trong objective space (khong qua decision
variables / optimizer) de co ground truth tuyet doi ro rang -- day la
cong cu kiem tra rieng cong thuc phan vung/severity/RHI (core/*,
detectors/*), khong phai benchmark DMOP that (do la viec cua E1 tren
CEC2018).

QUAN TRONG (sua sau lan chay dau): "affected" PHAI duoc dinh nghia bang
CHINH ham assign_regions() cua pipeline (core/partition.py), khong phai
mot nguong cosine doc lap -- neu khong, tap "affected" se khong khop
ranh gioi vung Das-Dennis that va se tran qua nhieu vung lan can, lam
E0 khong con la sanity check sach (da phat hien qua lan chay dau tien
FAIL: detector_correct chi 3/10).
"""
import numpy as np

from prdrl.core.partition import assign_regions


class ToyRegionalShift:
    """
    N nghiem phan bo tren simplex 2-muc-tieu f1+f2=1, f1,f2>=0 (Pareto
    front tinh, khong doi). Vung "affected" = DUNG MOT region (theo
    assign_regions voi cung V dung trong pipeline that) gan
    target_region_v nhat; CHI cac nghiem trong region do bi dich, moi
    nghiem khac giu nguyen tuyet doi (d_i = 0 -- ground truth).
    """

    def __init__(self, N=150, target_region_v=(0.75, 0.25), shift_magnitude=0.35,
                seed=0):
        self.N = N
        self.target_region_v = np.asarray(target_region_v, dtype=float)
        self.target_region_v = self.target_region_v / self.target_region_v.sum()
        self.shift_magnitude = float(shift_magnitude)
        self.rng = np.random.default_rng(seed)

    def sample_front(self, V):
        """
        F_minus: (N, 2), f1 uniform tren [0,1], f2 = 1 - f1.
        V: (R, 2) reference directions dung CHUNG voi pipeline -- can de
           xac dinh affected region bang chinh assign_regions().

        return: (F_minus, affected_mask, target_region_idx)
        """
        f1 = self.rng.uniform(0.02, 0.98, size=self.N)
        f2 = 1.0 - f1
        F_minus = np.column_stack([f1, f2])

        # F_minus_tilde xap xi = chinh no (front nam tren simplex, gan
        # dung thang chuan hoa [0,1] mac dinh) -- du de xac dinh region.
        r_of_i = assign_regions(F_minus, V)
        # target region = region gan target_region_v nhat theo cosine.
        v = self.target_region_v / (np.linalg.norm(self.target_region_v) + 1e-12)
        norms_v = np.linalg.norm(V, axis=1) + 1e-12
        cos_to_targets = (V @ v) / norms_v
        target_region_idx = int(np.argmax(cos_to_targets))

        affected_mask = (r_of_i == target_region_idx)
        return F_minus, affected_mask, target_region_idx

    def apply_change(self, F_minus, affected_mask):
        """F_plus = F_minus, TRU cac nghiem affected duoc cong them
        shift_magnitude*(1,1) (day PF cuc bo ra xa goc toa do -> chat
        luong xau di ro rang CHI cho vung do)."""
        F_plus = F_minus.copy()
        F_plus[affected_mask] += self.shift_magnitude
        return F_plus
