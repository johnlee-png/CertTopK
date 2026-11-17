"""Pre-pass action-sequence extraction: the K events immediately preceding
each pass, for the Sequence Transformer (NEXT_STEPS.md, Tier 2 direction #8,
"Sequence Transformer").

Reuses the same match-indexed event-stream structure as
src.data.turnover_labels (events_360_all.parquet, sorted by `index` within
each match), but looks *backward* from the pass instead of forward.
"""

import numpy as np
import pandas as pd

MAX_SEQ_LEN = 5
SEQ_FEATURE_DIM = 6  # [delta_t_norm, x_norm, y_norm, is_own_team, is_pass, is_shot_or_carry]


def _extract_xy(events: pd.DataFrame) -> tuple:
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
    return x, y


def build_sequence_tensors(
    passes: pd.DataFrame,
    all_events: pd.DataFrame,
    max_len: int = MAX_SEQ_LEN,
    max_delta_seconds: float = 60.0,
) -> tuple:
    """passes must supply: match_id, index, period, minute, second, team_name.
    all_events must supply: match_id, index, period, minute, second, team_name,
    type_name, location (same schema as events_360_all.parquet).

    Returns (seq, mask): seq is (n_passes, max_len, SEQ_FEATURE_DIM) float32,
    mask is (n_passes, max_len) bool. The most recent preceding event occupies
    the LAST slot (index max_len - 1); earlier slots are padding if fewer than
    max_len events are available (e.g., early in a match).
    """
    events = all_events[["match_id", "index", "period", "minute", "second", "team_name", "type_name", "location"]].copy()
    events["_abs_time"] = events["minute"] * 60 + events["second"]
    events["_x"], events["_y"] = _extract_xy(events)

    n = len(passes)
    seq = np.zeros((n, max_len, SEQ_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((n, max_len), dtype=bool)
    row_pos = {label: i for i, label in enumerate(passes.index)}

    for match_id, match_passes in passes.groupby("match_id"):
        match_events = events[events["match_id"] == match_id].sort_values("index")
        if match_events.empty:
            continue

        ev_index = match_events["index"].to_numpy()
        ev_period = match_events["period"].to_numpy()
        ev_time = match_events["_abs_time"].to_numpy()
        ev_team = match_events["team_name"].to_numpy()
        ev_type = match_events["type_name"].to_numpy()
        ev_x = match_events["_x"].to_numpy()
        ev_y = match_events["_y"].to_numpy()

        for idx, row in match_passes.iterrows():
            pos = np.searchsorted(ev_index, row["index"], side="left")
            start = max(0, pos - max_len)
            w_len = pos - start
            if w_len <= 0:
                continue

            abs_time_now = row["minute"] * 60 + row["second"]
            offset = max_len - w_len
            out_row = row_pos[idx]

            for j in range(w_len):
                k = start + j
                if ev_period[k] != row["period"]:
                    continue
                x, y = ev_x[k], ev_y[k]
                if np.isnan(x) or np.isnan(y):
                    continue

                delta_t = max(0.0, abs_time_now - ev_time[k])
                slot = offset + j
                is_own = 1.0 if ev_team[k] == row["team_name"] else 0.0
                t = ev_type[k]
                is_pass = 1.0 if t == "Pass" else 0.0
                is_shot_or_carry = 1.0 if t in ("Shot", "Carry") else 0.0

                seq[out_row, slot] = [
                    min(delta_t, max_delta_seconds) / max_delta_seconds,
                    x / 120.0, y / 80.0,
                    is_own, is_pass, is_shot_or_carry,
                ]
                mask[out_row, slot] = True

    return seq, mask
