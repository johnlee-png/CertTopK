"""un-xPass-style pass creativity metric (NEXT_STEPS.md Tier 2 direction #7,
"un-xPass CDR", after Robberechts et al., KDD 2023): creativity = originality
x value.

We approximate "originality" without a full choice-model: for each pass, we
rank all visible teammates in the freeze-frame by the xT of *their* position
(i.e. how threatening a target each teammate would have been), and measure
how far down that ranking the *actually chosen* receiver's position sits.
Passing to the most threatening-looking teammate is the "obvious" choice
(low originality); passing to a team-mate whose position looked comparatively
unthreatening -- yet the pass still had value -- is scored as creative.
"""

import numpy as np
import pandas as pd

from src.features.expected_threat import get_xt_value


def compute_creativity(
    passes: pd.DataFrame,
    xt_grid: np.ndarray,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
) -> pd.DataFrame:
    """passes must supply: freeze_frame, end_x, end_y, delta_xt, pass_success.

    Returns a DataFrame (aligned to passes.index) with columns:
    n_candidates, chosen_xt, best_candidate_xt, originality, creativity.
    """
    n = len(passes)
    n_candidates = np.zeros(n, dtype=int)
    chosen_xt = np.zeros(n)
    best_candidate_xt = np.zeros(n)
    originality = np.zeros(n)

    for i, (ff, end_x, end_y) in enumerate(
        zip(passes["freeze_frame"].values, passes["end_x"].values, passes["end_y"].values)
    ):
        if ff is None or len(ff) == 0:
            continue

        candidate_xts = []
        for p in ff:
            if not p.get("teammate", False) or p.get("actor", False):
                continue
            loc = p.get("location")
            if loc is None or len(loc) < 2:
                continue
            candidate_xts.append(get_xt_value(xt_grid, loc[0], loc[1], pitch_length, pitch_width))

        if not candidate_xts:
            continue

        candidate_xts = np.array(candidate_xts)
        chosen = get_xt_value(xt_grid, end_x, end_y, pitch_length, pitch_width)

        n_candidates[i] = len(candidate_xts)
        chosen_xt[i] = chosen
        best_candidate_xt[i] = candidate_xts.max()
        # fraction of visible teammates whose position was MORE threatening
        # than the one actually targeted -> how "unobvious" the choice was
        originality[i] = float((candidate_xts > chosen).mean())

    realized_value = passes["delta_xt"].values * passes["pass_success"].values
    creativity = originality * realized_value

    return pd.DataFrame(
        {
            "n_candidates": n_candidates,
            "chosen_xt": chosen_xt,
            "best_candidate_xt": best_candidate_xt,
            "originality": originality,
            "creativity": creativity,
        },
        index=passes.index,
    )
