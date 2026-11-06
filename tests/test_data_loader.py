"""Tests for the data cleaning and table reconstruction modules."""

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import flag_laps, build_tyre_stints, build_pit_stops


def test_flag_laps():
    sample_laps = pd.DataFrame([
        {
            "session_key": "2022_catalunya_R", "circuit_id": "catalunya", "driver_id": "VER",
            "lap_number": 1, "lap_time_seconds": 95.0, "track_status": "1", "compound": "SOFT",
            "is_pit_in_lap": False, "is_pit_out_lap": False
        },
        {
            "session_key": "2022_catalunya_R", "circuit_id": "catalunya", "driver_id": "VER",
            "lap_number": 2, "lap_time_seconds": 88.5, "track_status": "1", "compound": "SOFT",
            "is_pit_in_lap": False, "is_pit_out_lap": False
        },
        {
            "session_key": "2022_catalunya_R", "circuit_id": "catalunya", "driver_id": "VER",
            "lap_number": 3, "lap_time_seconds": 94.0, "track_status": "1", "compound": "SOFT",
            "is_pit_in_lap": True, "is_pit_out_lap": False
        },
        {
            "session_key": "2022_catalunya_R", "circuit_id": "catalunya", "driver_id": "VER",
            "lap_number": 4, "lap_time_seconds": 105.0, "track_status": "1", "compound": "MEDIUM",
            "is_pit_in_lap": False, "is_pit_out_lap": True
        },
        {
            "session_key": "2022_catalunya_R", "circuit_id": "catalunya", "driver_id": "VER",
            "lap_number": 5, "lap_time_seconds": 120.0, "track_status": "4", "compound": "MEDIUM",
            "is_pit_in_lap": False, "is_pit_out_lap": False
        }
    ])

    flagged = flag_laps(sample_laps)
    # Lap 1 is excluded (start lap)
    assert not flagged.loc[0, "is_clean_lap"]
    # Lap 2 is clean
    assert flagged.loc[1, "is_clean_lap"]
    # Lap 3 is pit in
    assert not flagged.loc[2, "is_clean_lap"]
    # Lap 4 is pit out
    assert not flagged.loc[3, "is_clean_lap"]
    # Lap 5 is Safety Car (code 4)
    assert not flagged.loc[4, "is_clean_lap"]

