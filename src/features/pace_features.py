"""Leakage-safe pace and race-context features.

Rule (plan section 5.3 / Stage A rule 7): every feature for lap t uses only information that
existed when lap t started. Rolling pace statistics are computed over the driver's previous
*clean* laps with ``shift(1)`` applied before ``rolling``. ``tests/test_feature_leakage.py``
verifies that features computed on a truncated race equal those computed on the full race.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.fuel_model import fuel_remaining_kg, lap_fraction

PACE_FEATURES = ["roll_mean_3", "roll_median_3", "last_lap_delta",
                 "laps_since_prev_clean", "n_prev_clean"]


def add_pace_features(laps: pd.DataFrame) -> pd.DataFrame:
    df = laps.copy()
    for c in PACE_FEATURES:
        df[c] = np.nan
    clean = df[df["is_clean_lap"]].sort_values(["session_key", "driver_id", "lap_number"])
    g = clean.groupby(["session_key", "driver_id"], sort=False)
    prev = g["lap_time_seconds"].shift(1)
    prev2 = g["lap_time_seconds"].shift(2)
    keys = [clean["session_key"], clean["driver_id"]]
    df.loc[clean.index, "roll_mean_3"] = prev.groupby(keys).transform(lambda s: s.rolling(3, min_periods=1).mean())
    df.loc[clean.index, "roll_median_3"] = prev.groupby(keys).transform(lambda s: s.rolling(3, min_periods=1).median())
    df.loc[clean.index, "last_lap_delta"] = prev - prev2
    df.loc[clean.index, "laps_since_prev_clean"] = clean["lap_number"] - g["lap_number"].shift(1)
    df.loc[clean.index, "n_prev_clean"] = g.cumcount()
    return df


def add_race_context(laps: pd.DataFrame) -> pd.DataFrame:
    """Plan table ``race_context``: state at the START of lap t (i.e. end of lap t-1)."""
    cfg = load_config()
    df = laps.copy()
    df["gap_to_ahead_end_s"] = np.nan
    df["gap_to_leader_end_s"] = np.nan
    done = df["lap_end_session_s"].notna()
    for (_, _), grp in df[done].groupby(["session_key", "lap_number"]):
        order = grp.sort_values("lap_end_session_s")
        t = order["lap_end_session_s"]
        df.loc[order.index, "gap_to_ahead_end_s"] = t.diff().values
        df.loc[order.index, "gap_to_leader_end_s"] = (t - t.iloc[0]).values

    df = df.sort_values(["session_key", "driver_id", "lap_number"])
    g = df.groupby(["session_key", "driver_id"], sort=False)
    df["gap_to_ahead_seconds"] = g["gap_to_ahead_end_s"].shift(1)
    df["gap_to_leader_seconds"] = g["gap_to_leader_end_s"].shift(1)
    df["position_at_start"] = g["position"].shift(1)
    df["is_in_dirty_air"] = df["gap_to_ahead_seconds"] <= cfg["traffic"]["dirty_air_gap_s"]
    df["laps_remaining"] = df["total_laps"] - (df["lap_number"] - 1)
    df["estimated_fuel_mass_kg"] = fuel_remaining_kg(df["lap_number"], df["total_laps"])
    df["fuel_proxy_lap_fraction"] = lap_fraction(df["lap_number"], df["total_laps"])
    status = df["track_status"].astype(str)
    df["active_race_control_status"] = np.select(
        [status.str.contains("5"), status.str.contains("4"), status.str.contains("[67]"), status.str.contains("2")],
        ["RED_FLAG", "SAFETY_CAR", "VSC", "YELLOW"], default="GREEN")
    return df.drop(columns=["gap_to_ahead_end_s", "gap_to_leader_end_s"])


def add_stint_features(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    df = laps.merge(stints[["session_key", "driver_id", "stint_number", "stint_start_lap"]],
                    on=["session_key", "driver_id", "stint_number"], how="left")
    df["stint_length_so_far"] = df["lap_number"] - df["stint_start_lap"] + 1
    return df

