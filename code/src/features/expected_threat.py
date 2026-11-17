"""Expected Threat (xT) grid computation for reward modeling."""

import numpy as np
import pandas as pd


def _extract_xy(series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract x, y arrays from a column of lists/arrays. Returns (x, y, valid_mask)."""
    def _safe_x(loc):
        try:
            if loc is not None and hasattr(loc, '__len__') and len(loc) >= 2:
                return float(loc[0])
        except (TypeError, ValueError):
            pass
        return np.nan

    def _safe_y(loc):
        try:
            if loc is not None and hasattr(loc, '__len__') and len(loc) >= 2:
                return float(loc[1])
        except (TypeError, ValueError):
            pass
        return np.nan

    x = series.apply(_safe_x).values
    y = series.apply(_safe_y).values
    valid = ~(np.isnan(x) | np.isnan(y))
    return x, y, valid


def build_xt_grid(
    events_df: pd.DataFrame,
    n_rows: int = 16,
    n_cols: int = 12,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
    n_iterations: int = 50,
    external_xg: np.ndarray | None = None,
) -> np.ndarray:
    """Build an Expected Threat grid from event data (vectorized).

    xT(z) = shot_prob(z) * avg_xG(z) + move_prob(z) * sum_z'[move_T(z,z') * xT(z')]

    By default avg_xG(z) is the raw empirical goals/shots ratio per zone,
    which is noisy in zones with few shots. If `external_xg` is given (an
    array aligned to `events_df`, e.g. StatsBomb's own `shot_statsbomb_xg`
    model), avg_xG(z) is instead the mean model-based xG of shots in that
    zone -- a calibrated alternative (NEXT_STEPS.md Tier 1, "xT calibration").
    We use StatsBomb's own pre-fit shot xG model here rather than an external
    provider (e.g. Understat) to avoid cross-source shot-matching error.
    """
    cell_w = pitch_length / n_cols
    cell_h = pitch_width / n_rows

    # Extract start locations
    loc_x, loc_y, loc_valid = _extract_xy(events_df["location"])

    # Compute grid cells for start locations
    col_s = np.clip((loc_x / cell_w).astype(int), 0, n_cols - 1)
    row_s = np.clip((loc_y / cell_h).astype(int), 0, n_rows - 1)

    type_arr = events_df["type_name"].values

    # === Shots ===
    shot_mask = loc_valid & (type_arr == "Shot")
    shot_count = np.zeros((n_rows, n_cols))
    goal_count = np.zeros((n_rows, n_cols))

    xg_sum = np.zeros((n_rows, n_cols))

    if shot_mask.any():
        np.add.at(shot_count, (row_s[shot_mask], col_s[shot_mask]), 1)

        goal_mask = shot_mask.copy()
        if "shot_outcome_name" in events_df.columns:
            outcomes = events_df["shot_outcome_name"].values
            goal_mask = shot_mask & (outcomes == "Goal")
        else:
            goal_mask = np.zeros_like(shot_mask)

        if goal_mask.any():
            np.add.at(goal_count, (row_s[goal_mask], col_s[goal_mask]), 1)

        if external_xg is not None:
            xg_valid = shot_mask & ~pd.isna(external_xg)
            if xg_valid.any():
                np.add.at(xg_sum, (row_s[xg_valid], col_s[xg_valid]), external_xg[xg_valid])

    # === Moves (Pass + Carry) ===
    move_type_mask = (type_arr == "Pass") | (type_arr == "Carry")

    # Get end locations: prefer pass_end_location, fallback to carry_end_location
    end_loc_series = pd.Series([None] * len(events_df), index=events_df.index)

    if "pass_end_location" in events_df.columns:
        pass_mask_col = events_df["pass_end_location"].notna()
        end_loc_series[pass_mask_col] = events_df.loc[pass_mask_col, "pass_end_location"]

    if "carry_end_location" in events_df.columns:
        carry_fill = end_loc_series.isna() & events_df["carry_end_location"].notna()
        end_loc_series[carry_fill] = events_df.loc[carry_fill, "carry_end_location"]

    end_x, end_y, end_valid = _extract_xy(end_loc_series)

    move_mask = loc_valid & move_type_mask & end_valid

    col_e = np.zeros(len(events_df), dtype=int)
    row_e = np.zeros(len(events_df), dtype=int)
    if end_valid.any():
        col_e[end_valid] = np.clip((end_x[end_valid] / cell_w).astype(int), 0, n_cols - 1)
        row_e[end_valid] = np.clip((end_y[end_valid] / cell_h).astype(int), 0, n_rows - 1)

    move_count = np.zeros((n_rows, n_cols))
    transition = np.zeros((n_rows, n_cols, n_rows, n_cols))

    if move_mask.any():
        np.add.at(move_count, (row_s[move_mask], col_s[move_mask]), 1)
        np.add.at(
            transition,
            (row_s[move_mask], col_s[move_mask], row_e[move_mask], col_e[move_mask]),
            1,
        )

    # Normalize
    total_actions = np.maximum(shot_count + move_count, 1)
    shot_prob = shot_count / total_actions
    move_prob = move_count / total_actions
    if external_xg is not None:
        avg_xg = np.divide(xg_sum, shot_count, out=np.zeros_like(xg_sum), where=shot_count > 0)
    else:
        avg_xg = np.divide(goal_count, shot_count, out=np.zeros_like(goal_count), where=shot_count > 0)

    # Normalize transition matrix
    move_count_exp = np.maximum(move_count[:, :, np.newaxis, np.newaxis], 1)
    trans_prob = transition / move_count_exp

    # Value iteration
    xt = np.zeros((n_rows, n_cols))
    for _ in range(n_iterations):
        move_value = np.sum(trans_prob * xt[np.newaxis, np.newaxis, :, :], axis=(2, 3))
        xt = shot_prob * avg_xg + move_prob * move_value

    return xt


def get_xt_value(
    xt_grid: np.ndarray,
    x: float,
    y: float,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
) -> float:
    """Look up xT value for a given pitch coordinate."""
    n_rows, n_cols = xt_grid.shape
    col = min(int(x / (pitch_length / n_cols)), n_cols - 1)
    row = min(int(y / (pitch_width / n_rows)), n_rows - 1)
    return float(xt_grid[max(0, row), max(0, col)])


def compute_delta_xt(
    df: pd.DataFrame,
    xt_grid: np.ndarray,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
) -> pd.Series:
    """Compute xT change (reward) for each pass (vectorized)."""
    n_rows, n_cols = xt_grid.shape
    cell_w = pitch_length / n_cols
    cell_h = pitch_width / n_rows

    s_col = np.clip((df["start_x"].values / cell_w).astype(int), 0, n_cols - 1)
    s_row = np.clip((df["start_y"].values / cell_h).astype(int), 0, n_rows - 1)
    e_col = np.clip((df["end_x"].values / cell_w).astype(int), 0, n_cols - 1)
    e_row = np.clip((df["end_y"].values / cell_h).astype(int), 0, n_rows - 1)

    start_xt = xt_grid[s_row, s_col]
    end_xt = xt_grid[e_row, e_col]

    return pd.Series(end_xt - start_xt, index=df.index)
