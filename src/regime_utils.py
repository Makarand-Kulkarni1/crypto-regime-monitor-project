"""
regime_utils.py
Shared regime-labeling rule. This is the SAME logic used in
03_regime_labeling_and_model.ipynb to create training labels - kept here as a
single source of truth so the live dashboard can correctly determine what
regime *actually* occurred at a given point in time (for accuracy tracking),
using the exact same rule and thresholds the model was trained against.
"""


def label_regime(vol_ratio: float, trend_strength: float,
                  vol_ratio_threshold: float, trend_strength_threshold: float) -> str:
    """
    Classify a single row's regime given its vol_ratio/trend_strength values
    and the quantile-derived thresholds saved alongside the trained model.
    """
    if vol_ratio >= vol_ratio_threshold:
        return "High-Vol Breakout"
    elif abs(trend_strength) >= trend_strength_threshold:
        return "Low-Vol Trending"
    else:
        return "Ranging / Choppy"
