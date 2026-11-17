"""Event-only baseline feature extraction."""

import numpy as np
import pandas as pd


def extract_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract event-only features for pass success prediction.

    These form the baseline (no 360 context) feature set.
    """
    features = pd.DataFrame(index=df.index)

    # Pass start/end coordinates
    features["start_x"] = df["start_x"]
    features["start_y"] = df["start_y"]
    features["end_x"] = df["end_x"]
    features["end_y"] = df["end_y"]

    # Pass geometry
    dx = df["end_x"] - df["start_x"]
    dy = df["end_y"] - df["start_y"]
    features["pass_length"] = np.sqrt(dx**2 + dy**2)
    features["pass_angle"] = np.arctan2(dy, dx)
    features["forward_distance"] = dx

    # Normalized coordinates (0-1 scale)
    features["start_x_norm"] = df["start_x"] / 120.0
    features["start_y_norm"] = df["start_y"] / 80.0
    features["end_x_norm"] = df["end_x"] / 120.0
    features["end_y_norm"] = df["end_y"] / 80.0

    # Distance to opponent goal from start and end
    features["start_dist_to_goal"] = np.sqrt(
        (120.0 - df["start_x"])**2 + (40.0 - df["start_y"])**2
    )
    features["end_dist_to_goal"] = np.sqrt(
        (120.0 - df["end_x"])**2 + (40.0 - df["end_y"])**2
    )

    # Match context
    if "minute" in df.columns:
        features["minute"] = df["minute"]
    if "period" in df.columns:
        features["period"] = df["period"]

    # Under pressure
    if "under_pressure" in df.columns:
        features["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    else:
        features["under_pressure"] = 0

    # Pass height
    if "pass_height_name" in df.columns:
        features["is_ground_pass"] = (df["pass_height_name"] == "Ground Pass").astype(int)
        features["is_high_pass"] = (df["pass_height_name"] == "High Pass").astype(int)
        features["is_low_pass"] = (df["pass_height_name"] == "Low Pass").astype(int)

    # Pass body part
    if "pass_body_part_name" in df.columns:
        features["is_right_foot"] = (df["pass_body_part_name"] == "Right Foot").astype(int)

    # Pitch zone indicators
    features["start_in_own_third"] = (df["start_x"] < 40).astype(int)
    features["start_in_mid_third"] = ((df["start_x"] >= 40) & (df["start_x"] < 80)).astype(int)
    features["start_in_final_third"] = (df["start_x"] >= 80).astype(int)
    features["end_in_final_third"] = (df["end_x"] >= 80).astype(int)
    features["end_in_penalty_area"] = (
        (df["end_x"] >= 102) & (df["end_y"] >= 18) & (df["end_y"] <= 62)
    ).astype(int)

    # Cross indicator
    if "pass_cross" in df.columns:
        features["is_cross"] = df["pass_cross"].fillna(False).astype(int)
    else:
        features["is_cross"] = 0

    # Switch indicator (large lateral movement)
    features["is_switch"] = (np.abs(dy) > 32).astype(int)

    return features


def extract_pass_outcome(df: pd.DataFrame) -> pd.Series:
    """Extract binary pass success label."""
    if "pass_outcome_name" in df.columns:
        # StatsBomb: outcome is null/NaN for successful passes
        return df["pass_outcome_name"].isna().astype(int)
    return pd.Series(1, index=df.index, name="pass_success")
