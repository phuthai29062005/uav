from dataclasses import dataclass
import numpy as np

@dataclass 
class RegionSnapshot:
    t: int
    region_id: int
    
    active: bool
    n: int
    
    X: np.ndarray
    Z: np.ndarray
    F: np.ndarray
    
    centroid_z: np.ndarray
    centroid_f: np.ndarray

class RegionHistory:
    def __init__(self, n_regions: int):
        self.n_regions = n_regions
        
        self._history = {
            r: {} for r in range(n_regions)
        }
    
    @staticmethod
    def _normalize_decision(X, xl, xu):
        
        X = np.asarray(X, dtype=float)
        xl = np.asarray(xl, dtype=float)
        xu = np.asarray(xu, dtype=float)
        
        span = xu - xl
        if np.any(span <= 0):
            raise ValueError("Upper and lower bounds must be different for normalization.")
        
        return (X - xl)/span
    
    def update(self, t, X, F, region_ids, xl, xu):
        
        X = np.asarray(X, dtype=float)
        F = np.asarray(F, dtype=float)
        region_ids = np.asarray(region_ids, dtype=int)
        xl = np.asarray(xl, dtype=float)
        xu = np.asarray(xu, dtype=float)
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if F.ndim != 2:
            raise ValueError("F must be a 2D array.")
        if region_ids.ndim != 1:
            raise ValueError("region_ids must be a 1D array.")
        if X.shape[0] != F.shape[0]:
            raise ValueError("The number of samples in X, F, and region_ids must be the same.")
        if len(region_ids) != X.shape[0]:
            raise ValueError("The length of region_ids must match the number of samples in X.")
        
        Z = self._normalize_decision(X, xl, xu)
        
        for r in range(self.n_regions):
            mask = region_ids == r
            
            Xr = X[mask]
            Zr = Z[mask]
            Fr = F[mask]
            
            nr = Xr.shape[0]
            if nr > 0:
                active = True
                
                centroid_z = np.mean(Zr, axis=0)
                centroid_f = np.mean(Fr, axis=0)
            else:
                active = False
                centroid_z = np.full(X.shape[1], np.nan)
                centroid_f = np.full(F.shape[1], np.nan)
                
            snapshot = RegionSnapshot(
                t=t,
                region_id=r,
                active=active,
                n=nr,
                X=Xr.copy(),
                Z=Zr.copy(),
                F=Fr.copy(),
                centroid_z=centroid_z.copy(),
                centroid_f=centroid_f.copy()
            )
            
            self._history[r][t] = snapshot
    
    def get_region(self, region_id, t):
        if region_id < 0 or region_id >= self.n_regions:
            raise ValueError(f"region_id must be in the range [0, {self.n_regions - 1}].")   
        
        return self._history[region_id].get(t, None)
    
    def get_centroid_sequence(self, region_id, end_t, length):
        if length <= 0:
            raise ValueError("length must be a positive integer.")
        
        start_t = end_t - length + 1
        centroids = []
        
        for t in range(start_t, end_t + 1):
            snapshot = self.get_region(region_id, t)
            
            if snapshot is None:
                return None
            
            if not snapshot.active:
                return None
            
            if np.any(np.isnan(snapshot.centroid_z)):
                return None
            
            centroids.append(snapshot.centroid_z)
        
        return np.stack(centroids, axis=0)
    