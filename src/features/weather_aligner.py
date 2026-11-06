"""Backward as-of alignment of weather observations onto laps (no look-ahead)."""

from __future__ import annotations

import pandas as pd

WEATHER_COLS = ["air_temperature", "track_temperature", "humidity", "wind_speed"]


def align_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest weather observation recorded at or before each lap's start.

    No backward-fill is applied: a lap that starts before the first observation keeps NaN.
    (In the 2022 races used here the weather stream starts more than an hour before lap 1.)
    """
    left = laps.sort_values("lap_start_session_s").copy()
    right = weather[["session_key", "obs_session_s"] + WEATHER_COLS].sort_values("obs_session_s")
    has_start = left["lap_start_session_s"].notna()
    merged = pd.merge_asof(
        left[has_start], right, left_on="lap_start_session_s", right_on="obs_session_s",
        by="session_key", direction="backward", allow_exact_matches=True,
    )
    out = pd.concat([merged, left[~has_start]], ignore_index=True)
    out["weather_age_s"] = out["lap_start_session_s"] - out["obs_session_s"]
    return out.sort_values(["session_key", "driver_id", "lap_number"]).reset_index(drop=True)

