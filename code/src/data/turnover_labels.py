"""Realized turnover-cost labeling: opponent xT reached shortly after a failed pass."""

import numpy as np
import pandas as pd


def _vectorized_xt(events: pd.DataFrame, xt_grid: np.ndarray, pitch_length: float = 120.0, pitch_width: float = 80.0) -> np.ndarray:
    """Vectorized xT lookup for each event's location. Missing/invalid location -> 0.0."""
    n_rows, n_cols = xt_grid.shape
    cell_w = pitch_length / n_cols
    cell_h = pitch_width / n_rows

    def _xy(loc):
        try:
            if loc is not None and hasattr(loc, "__len__") and len(loc) >= 2:
                return float(loc[0]), float(loc[1])
        except (TypeError, ValueError):
            pass
        return np.nan, np.nan

    xy = events["location"].apply(_xy)
    x = np.array([v[0] for v in xy])
    y = np.array([v[1] for v in xy])
    valid = ~(np.isnan(x) | np.isnan(y))

    col = np.clip((np.nan_to_num(x) / cell_w).astype(int), 0, n_cols - 1)
    row = np.clip((np.nan_to_num(y) / cell_h).astype(int), 0, n_rows - 1)
    xt = xt_grid[row, col]
    xt[~valid] = 0.0
    return xt


def compute_realized_turnover_cost(
    passes: pd.DataFrame,
    all_events: pd.DataFrame,
    xt_grid: np.ndarray,
    window_events: int = 10,
    window_seconds: float = 15.0,
) -> pd.Series:
    """For each failed pass (turnover), find the opponent's peak xT reached within
    the next `window_events` events (capped at `window_seconds`, same period only).

    Returns a Series aligned to `passes.index`. Passes that did not fail have no
    observed turnover consequence and are left as NaN.
    """
    events = all_events[["match_id", "index", "period", "minute", "second", "team_name", "location"]].copy()
    events["_abs_time"] = events["minute"] * 60 + events["second"]
    events["_xt"] = _vectorized_xt(events, xt_grid)

    cost = pd.Series(np.nan, index=passes.index)
    turnovers = passes[passes["pass_outcome_name"].notna()]

    for match_id, match_turnovers in turnovers.groupby("match_id"):
        match_events = events[events["match_id"] == match_id].sort_values("index")
        if match_events.empty:
            continue

        ev_index = match_events["index"].to_numpy()
        ev_period = match_events["period"].to_numpy()
        ev_time = match_events["_abs_time"].to_numpy()
        ev_team = match_events["team_name"].to_numpy()
        ev_xt = match_events["_xt"].to_numpy()

        for idx, row in match_turnovers.iterrows():
            pos = np.searchsorted(ev_index, row["index"], side="right")
            end = min(pos + window_events, len(ev_index))
            if pos >= end:
                cost.loc[idx] = 0.0
                continue

            same_period = ev_period[pos:end] == row["period"]
            within_time = (ev_time[pos:end] - (row["minute"] * 60 + row["second"])) <= window_seconds
            is_opponent = ev_team[pos:end] != row["team_name"]
            keep = same_period & within_time & is_opponent

            cost.loc[idx] = float(ev_xt[pos:end][keep].max()) if keep.any() else 0.0

    return cost
