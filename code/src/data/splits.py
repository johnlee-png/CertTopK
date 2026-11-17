"""Data split utilities for match-level train/valid/test splitting."""

import numpy as np
import pandas as pd


def match_level_split(
    df: pd.DataFrame,
    match_col: str = "match_id",
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame at match level to prevent data leakage.

    Returns (train_df, valid_df, test_df).
    """
    matches = df[match_col].unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(matches)

    n = len(matches)
    n_train = int(n * train_ratio)
    n_valid = int(n * (train_ratio + valid_ratio))

    train_matches = set(matches[:n_train])
    valid_matches = set(matches[n_train:n_valid])
    test_matches = set(matches[n_valid:])

    train_df = df[df[match_col].isin(train_matches)].copy()
    valid_df = df[df[match_col].isin(valid_matches)].copy()
    test_df = df[df[match_col].isin(test_matches)].copy()

    print(f"Split: {len(train_matches)} train / {len(valid_matches)} valid / {len(test_matches)} test matches")
    print(f"Passes: {len(train_df)} train / {len(valid_df)} valid / {len(test_df)} test")

    return train_df, valid_df, test_df
