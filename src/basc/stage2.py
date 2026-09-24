"""Stage 2: distribution-free binary isotonic recalibration.

Fits isotonic regression (PAVA) of binary agreement B on a frozen score U over
a held-out calibration split. Unseen scores are assigned the fitted value of
their nearest fitted knot -- the block-valued extension covered by the
binary-isotonic calibration bound of Rossellini et al. (2026). (sklearn's
default `.predict` interpolates linearly between knots and is not used.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class BinaryIsotonic:
    iso: IsotonicRegression
    n_train: int
    n_knots: int

    @property
    def knots_x(self) -> np.ndarray:
        return np.asarray(self.iso.X_thresholds_, dtype=float)

    @property
    def knots_y(self) -> np.ndarray:
        return np.asarray(self.iso.y_thresholds_, dtype=float)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Fitted value of the nearest knot (ties go to the lower knot)."""
        U = np.asarray(scores, dtype=float)
        kx, ky = self.knots_x, self.knots_y
        if kx.size == 1:
            out = np.full_like(U, float(ky[0]))
        else:
            idx_r = np.clip(np.searchsorted(kx, U), 1, kx.size - 1)
            idx_l = idx_r - 1
            pick_right = (kx[idx_r] - U) < (U - kx[idx_l])
            out = ky[np.where(pick_right, idx_r, idx_l)]
        return np.clip(out, 0.0, 1.0)


def fit_binary_isotonic(scores: np.ndarray, agreements: np.ndarray) -> BinaryIsotonic:
    """Fit U -> B on binary outcomes B in {0, 1}."""
    s = np.asarray(scores, dtype=float)
    a = np.asarray(agreements, dtype=float)
    if s.ndim != 1 or s.shape != a.shape:
        raise ValueError(f"scores/agreements must be matching 1D arrays; got {s.shape}, {a.shape}")
    if not np.isin(np.unique(a), (0.0, 1.0)).all():
        raise ValueError(f"agreements must be binary, got unique={np.unique(a)}")
    if s.shape[0] < 2:
        raise ValueError("need at least 2 points to fit isotonic regression")
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(s, a)
    return BinaryIsotonic(iso=iso, n_train=int(s.shape[0]), n_knots=int(len(iso.X_thresholds_)))
