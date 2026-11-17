"""Occlusion-aware imputation for 360 pressure/count features.

StatsBomb 360 only records players inside the camera's `visible_area` polygon;
players outside it are simply absent from `freeze_frame`, which the existing
pipeline implicitly treats as "not present" when computing pressure counts
(this is exactly the mechanism behind Theorem 3, Monotone Downward Bias, in
paper/main.tex). This module estimates how many of the up-to-11 opponents on
the pitch fall in the *invisible* region for a given pass, and assumes they
are spread uniformly over it, so pressure-count features can be corrected
upward instead of implicitly zero-imputed.
"""

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
N_OPPONENTS_ON_PITCH = 11

PITCH_POLYGON = Polygon([(0, 0), (PITCH_LENGTH, 0), (PITCH_LENGTH, PITCH_WIDTH), (0, PITCH_WIDTH)])


def _parse_visible_area(va) -> Polygon | None:
    try:
        if va is None or len(va) < 6:
            return None
        coords = list(zip(va[0::2], va[1::2]))
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        poly = poly.intersection(PITCH_POLYGON)
        return None if poly.is_empty else poly
    except (TypeError, ValueError):
        return None


def compute_occlusion_correction(
    visible_area,
    n_visible_opponents: int,
    point: tuple,
    radius: float,
) -> float:
    """Expected number of additional (unseen) opponents within `radius` of
    `point`, assuming missing opponents are uniformly spread over the
    camera's invisible region of the pitch. Returns 0.0 if visible_area is
    missing/degenerate or nobody is missing.
    """
    n_missing = max(0, N_OPPONENTS_ON_PITCH - n_visible_opponents)
    if n_missing == 0:
        return 0.0

    visible_poly = _parse_visible_area(visible_area)
    if visible_poly is None:
        return 0.0

    invisible = PITCH_POLYGON.difference(visible_poly)
    if invisible.is_empty or invisible.area <= 1e-6:
        return 0.0

    circle = Point(point).buffer(radius)
    overlap = circle.intersection(invisible).area
    return float(n_missing * (overlap / invisible.area))


def add_occlusion_corrected_features(passes: pd.DataFrame, X_full: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X_full with occlusion-corrected pressure-count columns.

    `passes` must supply (aligned to X_full's index): freeze_frame,
    visible_area, start_x, start_y, end_x, end_y.
    """
    def _n_visible_opp(ff):
        if ff is None:
            return 0
        return sum(1 for p in ff if not p.get("teammate", True))

    n_visible = passes["freeze_frame"].apply(_n_visible_opp)

    corr_receiver_3 = np.zeros(len(passes))
    corr_receiver_5 = np.zeros(len(passes))
    corr_passer_3 = np.zeros(len(passes))
    corr_passer_5 = np.zeros(len(passes))

    cols = zip(
        passes["visible_area"].values, n_visible.values,
        passes["start_x"].values, passes["start_y"].values,
        passes["end_x"].values, passes["end_y"].values,
    )
    for i, (va, nv, sx, sy, ex, ey) in enumerate(cols):
        corr_receiver_3[i] = compute_occlusion_correction(va, nv, (ex, ey), 3.0)
        corr_receiver_5[i] = compute_occlusion_correction(va, nv, (ex, ey), 5.0)
        corr_passer_3[i] = compute_occlusion_correction(va, nv, (sx, sy), 3.0)
        corr_passer_5[i] = compute_occlusion_correction(va, nv, (sx, sy), 5.0)

    X_imputed = X_full.copy()
    X_imputed["receiver_opp_3m"] = X_full["receiver_opp_3m"].values + corr_receiver_3
    X_imputed["receiver_opp_5m"] = X_full["receiver_opp_5m"].values + corr_receiver_5
    X_imputed["passer_opp_3m"] = X_full["passer_opp_3m"].values + corr_passer_3
    X_imputed["passer_opp_5m"] = X_full["passer_opp_5m"].values + corr_passer_5

    corrections = pd.DataFrame({
        "receiver_opp_3m_correction": corr_receiver_3,
        "receiver_opp_5m_correction": corr_receiver_5,
        "passer_opp_3m_correction": corr_passer_3,
        "passer_opp_5m_correction": corr_passer_5,
    }, index=passes.index)

    return X_imputed, corrections
