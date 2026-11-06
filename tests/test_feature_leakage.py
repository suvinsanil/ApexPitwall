"""Unit tests for leakage prevention in feature engineering.

Asserts that features computed at lap t using partial session data are identical
to those computed on full session data (zero lookahead leakage).
"""

import numpy as np
import pandas as pd
import pytest

from src.features.fuel_model import fuel_remaining_kg, lap_fraction
from src.features.pace_features import add_pace_features, add_race_context


def test_fuel_proxy_properties():
    # Lap 1 fuel must equal max_fuel_kg
    f1 = fuel_remaining_kg(1, 66)
    assert np.isclose(f1, 110.0)

    # Monotonically decreasing
    f_mid = fuel_remaining_kg(33, 66)
    f_end = fuel_remaining_kg(66, 66)
    assert f1 > f_mid > f_end
    assert f_end >= 0.0

    # Lap fraction check
    assert lap_fraction(1, 66) == 0.0
    assert np.isclose(lap_fraction(66, 66), 65.0 / 66.0)


def test_rolling_pace_no_lookahead():
    """Verify that lap t features do not change if future laps are modified or appended."""
    laps_data = []
    # Create 10 clean laps for 1 driver
    for lap in range(1, 11):
        laps_data.append({
            "session_key": "2022_catalunya_R",
            "driver_id": "VER",
            "lap_number": lap,
            "lap_time_seconds": 88.0 + 0.1 * lap,
            "is_clean_lap": True,
            "stint_number": 1
        })
    df_full = pd.DataFrame(laps_data)
    df_full_feat = add_pace_features(df_full)

    # Now take truncated dataset up to lap 5
    df_trunc = df_full.iloc[:5].copy()
    df_trunc_feat = add_pace_features(df_trunc)

    # Compare lap 1 to 5 features
    for col in ["roll_mean_3", "roll_median_3", "last_lap_delta"]:
        vals_full = df_full_feat.loc[:4, col].values
        vals_trunc = df_trunc_feat.loc[:4, col].values
        np.testing.assert_allclose(vals_full, vals_trunc, equal_nan=True)

    # Crucial: On lap 1, rolling mean must be NaN because shift(1) leaves no prior clean lap
    assert np.isnan(df_full_feat.loc[0, "roll_mean_3"])
    # On lap 2, rolling mean of prior 1 lap should equal lap 1 lap_time
    assert np.isclose(df_full_feat.loc[1, "roll_mean_3"], df_full.loc[0, "lap_time_seconds"])
