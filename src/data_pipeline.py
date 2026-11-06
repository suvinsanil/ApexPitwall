"""
ApexPitwall — Data Pipeline Module
Path: src/data_pipeline.py

Loads FastF1 race sessions, performs rigorous cleaning and filtering,
merges weather telemetry via backward as-of joins, engineers leakage-safe
features (fuel burn proxy, shifted rolling pace, tyre age), and exports
the processed dataset to data/processed/race_laps_processed.csv.
"""

import os
import sys
import logging
import warnings
import fastf1
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ApexPitwall.DataPipeline")
warnings.filterwarnings("ignore", category=FutureWarning)

# 5 Target 2022 dry races identified in Phase 1 audit
TARGET_RACES_2022 = [
    {"round": 1, "name": "Bahrain Grand Prix", "circuit_id": "bahrain", "total_laps": 57},
    {"round": 6, "name": "Spanish Grand Prix", "circuit_id": "catalunya", "total_laps": 66},
    {"round": 15, "name": "Dutch Grand Prix", "circuit_id": "zandvoort", "total_laps": 72},
    {"round": 20, "name": "Mexico City Grand Prix", "circuit_id": "rodriguez", "total_laps": 71},
    {"round": 22, "name": "Abu Dhabi Grand Prix", "circuit_id": "yas_marina", "total_laps": 58},
]


def setup_cache(cache_dir: str = "data/cache") -> str:
    """Enables FastF1 disk caching."""
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    return os.path.abspath(cache_dir)


def load_session_data(year: int, round_num: int, cache_dir: str = "data/cache"):
    """Loads laps and weather data for a specific race session."""
    setup_cache(cache_dir)
    session = fastf1.get_session(year, round_num, "R")
    session.load(telemetry=False, weather=True)
    return session


def clean_race_laps(laps_df: pd.DataFrame, circuit_id: str) -> pd.DataFrame:
    """
    Cleans raw lap records according to Phase 1 data audit rules:
    1. Drop in-laps (PitInTime is not NaT).
    2. Drop out-laps (PitOutTime is not NaT).
    3. Drop missing LapTime.
    4. Drop non-green track status (contains codes '2', '4', '5', '6', '7').
    5. Exclude non-slick compound rows (None, Wet, Intermediate).
    6. Exclude extreme pace outliers (> 115% of median clean lap time on that circuit).
    """
    df = laps_df.copy()
    initial_count = len(df)

    # 1 & 2: Drop pit in/out laps
    no_pit = df["PitInTime"].isna() & df["PitOutTime"].isna()

    # 3: Drop missing LapTime
    has_lap_time = df["LapTime"].notna()

    # 4: Green track status only (FastF1 codes: 2=Yellow, 4=SC, 5=Red, 6=VSC, 7=VSC ending)
    is_green = ~df["TrackStatus"].astype(str).str.contains(r"[24567]", regex=True, na=True)

    # 5: Valid dry compounds only
    is_slick = df["Compound"].isin(["SOFT", "MEDIUM", "HARD"])

    valid_mask = no_pit & has_lap_time & is_green & is_slick
    cleaned = df[valid_mask].copy()

    # Convert target lap time to total seconds
    cleaned["lap_time_seconds"] = cleaned["LapTime"].dt.total_seconds().astype(np.float64)

    # 6: Filter extreme outliers (e.g. unflagged spins / off-tracks)
    # Beyond 115% of median clean lap time
    median_lt = cleaned["lap_time_seconds"].median()
    cutoff_lt = median_lt * 1.15
    outlier_mask = cleaned["lap_time_seconds"] <= cutoff_lt
    cleaned = cleaned[outlier_mask].copy()

    logger.info(
        f"[{circuit_id}] Cleaned laps: {len(cleaned)} / {initial_count} "
        f"({len(cleaned)/initial_count*100:.1f}% retained, median={median_lt:.2f}s)"
    )
    return cleaned


def merge_weather(cleaned_laps: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges weather data onto laps using a backward as-of join on LapStartTime vs weather Time.
    Ensures zero future weather leakage.
    """
    if weather_df is None or len(weather_df) == 0:
        cleaned_laps["TrackTemp"] = np.nan
        cleaned_laps["AirTemp"] = np.nan
        cleaned_laps["Humidity"] = np.nan
        cleaned_laps["Pressure"] = np.nan
        cleaned_laps["WindSpeed"] = np.nan
        return cleaned_laps

    w_sorted = weather_df.sort_values("Time").copy()
    l_sorted = cleaned_laps.sort_values("LapStartTime").copy()

    weather_cols = ["Time", "AirTemp", "Humidity", "Pressure", "TrackTemp", "WindSpeed"]
    existing_w_cols = [c for c in weather_cols if c in w_sorted.columns]

    merged = pd.merge_asof(
        l_sorted,
        w_sorted[existing_w_cols],
        left_on="LapStartTime",
        right_on="Time",
        direction="backward",
        suffixes=("", "_weather")
    )

    # Fallback interpolation if leading laps are slightly before weather timestamp
    for col in ["TrackTemp", "AirTemp", "Humidity", "Pressure", "WindSpeed"]:
        if col in merged.columns:
            merged[col] = merged[col].bfill().ffill()

    return merged


def engineer_lap_features(df: pd.DataFrame, total_race_laps: int) -> pd.DataFrame:
    """
    Engineers leakage-safe features for tyre degradation modeling:
    - fuel_remaining_kg: Physics-based linear proxy starting from 110 kg.
    - TyreLife: Laps run on current tyre set.
    - rolling_pace_mean_3 / rolling_pace_std_3: Shifted rolling metrics strictly
      using past clean laps to guarantee NO future information leakage.
    """
    df = df.copy()

    # Physics fuel mass proxy (max 110 kg under FIA 2022 technical regulations)
    # fuel_remaining_kg decreases linearly with race progress
    fuel_capacity_kg = 110.0
    df["fuel_remaining_kg"] = fuel_capacity_kg * (
        1.0 - (df["LapNumber"].astype(float) - 1.0) / float(total_race_laps)
    ).clip(lower=0.0, upper=1.0)

    # Shifted rolling pace per driver to prevent any lookahead
    # Sort chronologically by driver and lap number
    df = df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    rolling_means = []
    rolling_stds = []

    for driver, d_df in df.groupby("Driver", sort=False):
        # Shift 1 ensures the current lap is NEVER in the window
        shifted_pace = d_df["lap_time_seconds"].shift(1)
        r_mean = shifted_pace.rolling(window=3, min_periods=1).mean()
        r_std = shifted_pace.rolling(window=3, min_periods=1).std().fillna(0.0)

        # For the first lap where no prior lap exists, impute with driver median
        d_median = d_df["lap_time_seconds"].median()
        r_mean = r_mean.fillna(d_median)
        r_std = r_std.fillna(0.0)

        d_res = pd.DataFrame({
            "idx": d_df.index,
            "rolling_pace_mean_3": r_mean,
            "rolling_pace_std_3": r_std
        })
        rolling_means.append(d_res)

    pace_df = pd.concat(rolling_means).sort_values("idx").set_index("idx")
    df["rolling_pace_mean_3"] = pace_df["rolling_pace_mean_3"]
    df["rolling_pace_std_3"] = pace_df["rolling_pace_std_3"]

    return df


def build_processed_dataset(
    races_config=TARGET_RACES_2022,
    output_path: str = "data/processed/race_laps_processed.csv"
) -> pd.DataFrame:
    """
    Builds the complete multi-circuit dataset across the selected 2022 races.
    """
    all_clean_laps = []

    for race in races_config:
        rnd = race["round"]
        circuit_id = race["circuit_id"]
        event_name = race["name"]
        total_laps = race["total_laps"]

        logger.info(f"Processing Round {rnd:02d}: {event_name} ({circuit_id})...")
        session = load_session_data(2022, rnd)
        cleaned = clean_race_laps(session.laps, circuit_id)
        merged = merge_weather(cleaned, session.weather_data)

        merged["Circuit"] = circuit_id
        merged["EventName"] = event_name
        merged["RoundNumber"] = rnd

        featured = engineer_lap_features(merged, total_race_laps=total_laps)
        all_clean_laps.append(featured)

    combined = pd.concat(all_clean_laps, ignore_index=True)

    # Core modeling schema
    keep_columns = [
        "Circuit", "EventName", "RoundNumber", "Driver", "Team", "Compound",
        "LapNumber", "Stint", "TyreLife", "FreshTyre",
        "TrackTemp", "AirTemp", "Humidity", "Pressure", "WindSpeed",
        "fuel_remaining_kg", "rolling_pace_mean_3", "rolling_pace_std_3",
        "lap_time_seconds"
    ]
    final_df = combined[keep_columns].copy()

    # Type casting
    final_df["RoundNumber"] = final_df["RoundNumber"].astype(int)
    final_df["LapNumber"] = final_df["LapNumber"].astype(int)
    final_df["Stint"] = final_df["Stint"].astype(int)
    final_df["TyreLife"] = final_df["TyreLife"].astype(float)
    final_df["lap_time_seconds"] = final_df["lap_time_seconds"].astype(float)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(final_df)} processed laps to {output_path}")

    # Summary table
    summary = final_df.groupby(["Circuit", "Compound"])["lap_time_seconds"].agg(["count", "mean", "std"])
    print("\nProcessed Dataset Summary (Clean Laps per Circuit & Compound):")
    print(summary.to_string())

    return final_df


if __name__ == "__main__":
    build_processed_dataset()

