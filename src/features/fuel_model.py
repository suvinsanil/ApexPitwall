"""Mechanistic fuel-mass proxy.

Actual per-car fuel loads are not published. ASSUMPTION (configs/base_config.yaml): every
car starts with the 2022 regulation maximum (110 kg) and burns it linearly to zero at the
finish. The *time* effect of fuel is not assumed: it is estimated from data in Stage B
(src/models/degradation.py).
"""

from __future__ import annotations

import numpy as np

from src.config import load_config


def fuel_remaining_kg(lap_number, total_laps):
    """Fuel mass at the START of ``lap_number`` (1-indexed)."""
    max_fuel = load_config()["fuel"]["max_fuel_kg"]
    frac = (np.asarray(lap_number, dtype=float) - 1.0) / float(total_laps)
    return max_fuel * np.clip(1.0 - frac, 0.0, 1.0)


def lap_fraction(lap_number, total_laps):
    """Plan feature ``fuel_proxy_lap_fraction`` = (t - 1) / TotalLaps."""
    return (np.asarray(lap_number, dtype=float) - 1.0) / float(total_laps)

