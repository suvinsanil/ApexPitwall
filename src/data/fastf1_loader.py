"""Loads cached FastF1 race sessions into the plan's tabular schema.

Only fields that exist in the FastF1 feed are used. Stationary (wheel-stop) duration is NOT
in the public feed, so ``pit_stops.stationary_duration_seconds`` is always NaN (see Phase 1
open question 1 in PROJECT_OVERVIEW.md, now answered: not available).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import load_config, path

warnings.filterwarnings("ignore", category=FutureWarning)


def _seconds(s: pd.Series) -> pd.Series:
    return s.dt.total_seconds() if hasattr(s, "dt") else s.astype(float)


@dataclass
class RaceData:
    circuit_id: str
    session_key: str
    total_laps: int
    laps: pd.DataFrame       # race_laps grain (all laps, no filtering)
    weather: pd.DataFrame    # weather grain
    results: pd.DataFrame    # classification


def _enable_cache() -> None:
    import fastf1

    cache_dir = path("cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    fastf1.set_log_level(logging.WARNING)


def load_race(circuit_id: str) -> RaceData:
    """Load one race (laps, weather, results) from the FastF1 cache, downloading if needed."""
    import fastf1

    _enable_cache()
    cfg = load_config()
    info = cfg["circuit_info"][circuit_id]
    session = fastf1.get_session(cfg["season"], int(info["round"]), "R")
    session.load(telemetry=False, weather=True, messages=False)

    total_laps = int(session.total_laps)
    if total_laps != int(info["total_laps"]):
        logging.warning("%s: FastF1 total_laps=%s differs from config %s; using FastF1.",
                        circuit_id, total_laps, info["total_laps"])

    session_key = f"{cfg['season']}_{circuit_id}_R"
    raw = session.laps
    laps = pd.DataFrame({
        "session_key": session_key,
        "circuit_id": circuit_id,
        "driver_id": raw["Driver"].astype(str),
        "constructor_id": raw["Team"].astype(str),
        "lap_number": raw["LapNumber"].astype(int),
        "lap_time_seconds": _seconds(raw["LapTime"]),
        "sector_1_seconds": _seconds(raw["Sector1Time"]),
        "sector_2_seconds": _seconds(raw["Sector2Time"]),
        "sector_3_seconds": _seconds(raw["Sector3Time"]),
        "lap_start_session_s": _seconds(raw["LapStartTime"]),
        "lap_end_session_s": _seconds(raw["Time"]),
        "pit_in_session_s": _seconds(raw["PitInTime"]),
        "pit_out_session_s": _seconds(raw["PitOutTime"]),
        "stint_number": raw["Stint"],
        "compound": raw["Compound"].astype(object),
        "tyre_life": raw["TyreLife"].astype(float),
        "fresh_tyre": raw["FreshTyre"],
        "track_status": raw["TrackStatus"].astype(str),
        "position": raw["Position"],
        "deleted": raw["Deleted"] if "Deleted" in raw.columns else np.nan,
        "total_laps": total_laps,
    })
    laps["is_pit_in_lap"] = laps["pit_in_session_s"].notna()
    laps["is_pit_out_lap"] = laps["pit_out_session_s"].notna()
    laps = laps.sort_values(["driver_id", "lap_number"]).reset_index(drop=True)

    w = session.weather_data
    weather = pd.DataFrame({
        "session_key": session_key,
        "circuit_id": circuit_id,
        "obs_session_s": _seconds(w["Time"]),
        "air_temperature": w["AirTemp"].astype(float),
        "track_temperature": w["TrackTemp"].astype(float),
        "humidity": w["Humidity"].astype(float),
        "wind_speed": w["WindSpeed"].astype(float),
        "wind_direction": w["WindDirection"].astype(float),
        "rainfall": w["Rainfall"].astype(bool),
    }).sort_values("obs_session_s").reset_index(drop=True)

    r = session.results
    results = pd.DataFrame({
        "session_key": session_key,
        "circuit_id": circuit_id,
        "driver_id": r["Abbreviation"].astype(str),
        "constructor_id": r["TeamName"].astype(str),
        "classified_position": r["Position"],
        "status": r["Status"].astype(str),
    }).reset_index(drop=True)

    return RaceData(circuit_id, session_key, total_laps, laps, weather, results)

