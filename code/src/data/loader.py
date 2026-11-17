"""StatsBomb Open Data loader with 360 freeze-frame support."""

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm


def load_competitions(base_path: str) -> pd.DataFrame:
    """Load competition metadata."""
    path = Path(base_path) / "competitions.json"
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_matches(base_path: str, competition_id: int, season_id: int) -> pd.DataFrame:
    """Load match metadata for a given competition and season."""
    path = Path(base_path) / "matches" / str(competition_id) / f"{season_id}.json"
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_events(base_path: str, match_id: int) -> pd.DataFrame:
    """Load event data for a single match."""
    path = Path(base_path) / "events" / f"{match_id}.json"
    with open(path) as f:
        data = json.load(f)
    return pd.json_normalize(data, sep="_")


def load_lineups(base_path: str, match_id: int) -> pd.DataFrame:
    """Load lineup data for a single match."""
    path = Path(base_path) / "lineups" / f"{match_id}.json"
    with open(path) as f:
        data = json.load(f)
    records = []
    for team in data:
        team_name = team["team_name"]
        team_id = team["team_id"]
        for player in team["lineup"]:
            player["team_name"] = team_name
            player["team_id"] = team_id
            records.append(player)
    return pd.DataFrame(records)


def load_three_sixty(base_path: str, match_id: int) -> Optional[pd.DataFrame]:
    """Load 360 freeze-frame data for a single match. Returns None if not available."""
    path = Path(base_path) / "three-sixty" / f"{match_id}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not data:
            return None
        return pd.json_normalize(data, sep="_")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Warning: Skipping match {match_id} (corrupt 360 JSON: {e})")
        return None


def get_matches_with_360(base_path: str) -> list[int]:
    """Find all match IDs that have 360 data available."""
    three_sixty_dir = Path(base_path) / "three-sixty"
    if not three_sixty_dir.exists():
        return []
    return sorted(int(f.stem) for f in three_sixty_dir.glob("*.json"))


def load_all_competitions_with_360(base_path: str) -> pd.DataFrame:
    """Load all competitions and identify which matches have 360 data."""
    competitions = load_competitions(base_path)
    match_ids_with_360 = set(get_matches_with_360(base_path))

    all_matches = []
    for _, comp in competitions.iterrows():
        try:
            matches = load_matches(base_path, comp["competition_id"], comp["season_id"])
            matches["competition_name"] = comp["competition_name"]
            matches["season_name"] = comp["season_name"]
            matches["has_360"] = matches["match_id"].isin(match_ids_with_360)
            all_matches.append(matches)
        except FileNotFoundError:
            continue

    if not all_matches:
        return pd.DataFrame()
    return pd.concat(all_matches, ignore_index=True)


def load_and_join_events_360(
    base_path: str,
    match_ids: list[int],
    verbose: bool = True,
) -> pd.DataFrame:
    """Load events and join with 360 data for multiple matches.

    Returns a single DataFrame with events joined to their freeze-frame data.
    Only includes matches that have 360 data.
    """
    all_frames = []
    iterator = tqdm(match_ids, desc="Loading matches") if verbose else match_ids

    for match_id in iterator:
        events = load_events(base_path, match_id)
        events["match_id"] = match_id

        three_sixty = load_three_sixty(base_path, match_id)
        if three_sixty is None:
            continue

        # Join on event UUID
        merged = events.merge(
            three_sixty,
            left_on="id",
            right_on="event_uuid",
            how="inner",
            suffixes=("", "_360"),
        )
        all_frames.append(merged)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def filter_open_play_passes(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to open-play footed passes only, excluding set pieces."""
    mask = df["type_name"] == "Pass"

    # Exclude set piece passes
    if "pass_type_name" in df.columns:
        set_piece_types = {"Corner", "Free Kick", "Goal Kick", "Throw-in", "Kick Off"}
        mask &= ~df["pass_type_name"].isin(set_piece_types)

    # Exclude passes from set piece play patterns
    if "play_pattern_name" in df.columns:
        excluded_patterns = {
            "From Corner", "From Free Kick", "From Goal Kick",
            "From Throw In", "From Kick Off",
        }
        mask &= ~df["play_pattern_name"].isin(excluded_patterns)

    # Footed passes only (exclude head, keeper throws, etc.)
    if "pass_body_part_name" in df.columns:
        footed = {"Right Foot", "Left Foot"}
        mask &= df["pass_body_part_name"].isin(footed)

    return df[mask].copy()


def label_progressive_passes(
    df: pd.DataFrame,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
    min_forward_distance: float = 10.0,
    min_goal_approach_pct: float = 0.25,
) -> pd.DataFrame:
    """Label passes as progressive based on forward distance or goal approach.

    A pass is progressive if EITHER:
    - It moves the ball forward by at least min_forward_distance yards, OR
    - It brings the ball at least min_goal_approach_pct closer to the opponent goal.
    """
    df = df.copy()

    # Extract start/end coordinates (handles list, numpy array, and other iterables)
    def _get_coord(loc, idx):
        try:
            if loc is not None and hasattr(loc, '__len__') and len(loc) > idx:
                return float(loc[idx])
        except (TypeError, ValueError, IndexError):
            pass
        return None

    df["start_x"] = df["location"].apply(lambda loc: _get_coord(loc, 0))
    df["start_y"] = df["location"].apply(lambda loc: _get_coord(loc, 1))
    df["end_x"] = df["pass_end_location"].apply(lambda loc: _get_coord(loc, 0))
    df["end_y"] = df["pass_end_location"].apply(lambda loc: _get_coord(loc, 1))

    # Drop rows with missing coordinates
    df = df.dropna(subset=["start_x", "start_y", "end_x", "end_y"])

    # Forward distance (toward opponent goal at x=120)
    df["forward_distance"] = df["end_x"] - df["start_x"]

    # Distance to goal center (goal at x=pitch_length, y=pitch_width/2)
    goal_y = pitch_width / 2
    df["start_dist_to_goal"] = ((pitch_length - df["start_x"]) ** 2 + (goal_y - df["start_y"]) ** 2) ** 0.5
    df["end_dist_to_goal"] = ((pitch_length - df["end_x"]) ** 2 + (goal_y - df["end_y"]) ** 2) ** 0.5
    df["goal_approach_pct"] = (df["start_dist_to_goal"] - df["end_dist_to_goal"]) / df["start_dist_to_goal"].clip(lower=1e-6)

    # Progressive pass label
    df["is_progressive"] = (
        (df["forward_distance"] >= min_forward_distance) |
        (df["goal_approach_pct"] >= min_goal_approach_pct)
    )

    return df
