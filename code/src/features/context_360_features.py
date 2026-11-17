"""360 freeze-frame context feature extraction."""

import numpy as np
import pandas as pd


def _parse_freeze_frame(ff) -> list[dict]:
    """Parse freeze frame data from various formats (list, numpy array, string)."""
    if ff is None:
        return []
    if isinstance(ff, np.ndarray):
        return list(ff)
    if isinstance(ff, list):
        return ff
    if isinstance(ff, str):
        import json
        try:
            return json.loads(ff)
        except (json.JSONDecodeError, TypeError):
            return []
    if hasattr(ff, '__iter__'):
        return list(ff)
    return []


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return np.sqrt(np.sum((p1 - p2) ** 2))


def _count_players_in_radius(
    center: np.ndarray,
    players: list[dict],
    radius: float,
    teammate: bool | None = None,
) -> int:
    """Count players within a radius of center point."""
    count = 0
    for p in players:
        if teammate is not None and p.get("teammate") != teammate:
            continue
        loc = p.get("location")
        if loc is None:
            continue
        pos = np.array(loc)
        if _distance(center, pos) <= radius:
            count += 1
    return count


_MAX_PITCH_DIST = np.sqrt(120.0**2 + 80.0**2)  # ~144.2 yards


def _nearest_defender_distance(
    point: np.ndarray, players: list[dict],
) -> float:
    """Distance to nearest opponent from a given point. Capped at pitch diagonal."""
    min_dist = _MAX_PITCH_DIST
    for p in players:
        if p.get("teammate") is not False:
            continue
        loc = p.get("location")
        if loc is None:
            continue
        d = _distance(point, np.array(loc))
        if d < min_dist:
            min_dist = d
    return min_dist


def _count_defenders_in_lane(
    start: np.ndarray,
    end: np.ndarray,
    players: list[dict],
    lane_half_width: float = 2.0,
) -> int:
    """Count opponents within the passing lane corridor."""
    lane_vec = end - start
    lane_len = np.linalg.norm(lane_vec)
    if lane_len < 1e-6:
        return 0

    lane_unit = lane_vec / lane_len
    lane_normal = np.array([-lane_unit[1], lane_unit[0]])

    count = 0
    for p in players:
        if p.get("teammate", True):
            continue
        loc = p.get("location")
        if loc is None:
            continue
        pos = np.array(loc)
        rel = pos - start
        proj = np.dot(rel, lane_unit)
        if proj < 0 or proj > lane_len:
            continue
        perp = abs(np.dot(rel, lane_normal))
        if perp <= lane_half_width:
            count += 1
    return count


def _count_goalside_defenders(
    receiver_pos: np.ndarray,
    players: list[dict],
    goal_x: float = 120.0,
) -> int:
    """Count defenders between receiver and opponent goal."""
    count = 0
    for p in players:
        if p.get("teammate", True):
            continue
        loc = p.get("location")
        if loc is None:
            continue
        pos = np.array(loc)
        if pos[0] > receiver_pos[0] and pos[0] < goal_x:
            count += 1
    return count


def extract_360_features(df: pd.DataFrame, lane_half_width: float = 2.0) -> pd.DataFrame:
    """Extract 360 context features from freeze-frame data.

    Expects df to have columns: start_x, start_y, end_x, end_y, freeze_frame.
    """
    records = []

    for idx, row in df.iterrows():
        ff = _parse_freeze_frame(row.get("freeze_frame", []))
        start = np.array([row["start_x"], row["start_y"]])
        end = np.array([row["end_x"], row["end_y"]])

        feat = {}

        # === Passer-related features ===
        feat["passer_opp_3m"] = _count_players_in_radius(start, ff, 3.0, teammate=False)
        feat["passer_opp_5m"] = _count_players_in_radius(start, ff, 5.0, teammate=False)
        feat["passer_nearest_def_dist"] = _nearest_defender_distance(start, ff)
        feat["passer_teammate_5m"] = _count_players_in_radius(start, ff, 5.0, teammate=True)

        # Passer local numerical superiority (teammates - opponents in 5m)
        feat["passer_local_superiority"] = (
            feat["passer_teammate_5m"] - feat["passer_opp_5m"]
        )

        # === Receiver-related features ===
        feat["receiver_opp_3m"] = _count_players_in_radius(end, ff, 3.0, teammate=False)
        feat["receiver_opp_5m"] = _count_players_in_radius(end, ff, 5.0, teammate=False)
        feat["receiver_nearest_def_dist"] = _nearest_defender_distance(end, ff)
        feat["receiver_teammate_5m"] = _count_players_in_radius(end, ff, 5.0, teammate=True)
        feat["receiver_goalside_defenders"] = _count_goalside_defenders(end, ff)

        # Receiver free-space score: distance to nearest defender (capped at 20)
        feat["receiver_free_space"] = min(feat["receiver_nearest_def_dist"], 20.0) / 20.0

        # Cap raw distances for model stability
        feat["passer_nearest_def_dist"] = min(feat["passer_nearest_def_dist"], 50.0)
        feat["receiver_nearest_def_dist"] = min(feat["receiver_nearest_def_dist"], 50.0)

        # === Passing lane features ===
        feat["lane_defender_count"] = _count_defenders_in_lane(start, end, ff, lane_half_width)
        lane_len = max(np.linalg.norm(end - start), 1e-6)
        feat["lane_obstruction_ratio"] = feat["lane_defender_count"] / max(lane_len / 5.0, 1.0)

        # Corridor density: opponents per unit length in lane
        feat["corridor_density"] = feat["lane_defender_count"] / (lane_len / 10.0 + 1e-6)

        # === Visibility features ===
        visible_teammates = sum(1 for p in ff if p.get("teammate") is True)
        visible_opponents = sum(1 for p in ff if p.get("teammate") is False)
        feat["visible_teammates"] = visible_teammates
        feat["visible_opponents"] = visible_opponents
        feat["total_visible_players"] = visible_teammates + visible_opponents
        feat["context_completeness"] = min(len(ff) / 22.0, 1.0)

        # === Tactical interpretation features ===
        # Pressure escape: passer under pressure but receiver in space
        feat["pressure_escape"] = (
            1 if feat["passer_opp_3m"] >= 1 and feat["receiver_opp_3m"] == 0 else 0
        )

        # Between-lines reception proxy: receiver has defenders both ahead and behind
        opponents_behind = sum(
            1 for p in ff
            if not p.get("teammate", True)
            and p.get("location") is not None
            and p["location"][0] < end[0]
        )
        feat["between_lines_proxy"] = (
            1 if opponents_behind > 0 and feat["receiver_goalside_defenders"] > 0 else 0
        )

        # Forward angle openness: ratio of forward open space
        feat["forward_angle_openness"] = max(
            feat["receiver_nearest_def_dist"] - feat["passer_nearest_def_dist"], -10
        ) / 10.0

        # Local support options: teammates within 10m of receiver
        feat["receiver_support_count"] = _count_players_in_radius(end, ff, 10.0, teammate=True)

        records.append(feat)

    return pd.DataFrame(records, index=df.index)
