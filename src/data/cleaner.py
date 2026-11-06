"""Lap eligibility flags and stint / pit-stop table reconstruction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config


def flag_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Add boolean eligibility flags; ``is_clean_lap`` is the conjunction used for modelling.

    The outlier rule uses the median of laps that already pass every other rule in the same
    race. That median is a race-level descriptive statistic used only to discard obviously
    broken laps (spins, unflagged incidents); it is not a model feature.
    """
    cfg = load_config()["cleaning"]
    df = laps.copy()
    codes = "[" + cfg["non_green_codes"] + "]"
    df["is_non_green"] = df["track_status"].astype(str).str.contains(codes, regex=True, na=True)
    df["is_slick"] = df["compound"].isin(cfg["dry_compounds"])
    df["has_lap_time"] = df["lap_time_seconds"].notna()
    df["is_lap_one"] = df["lap_number"] == 1

    base = (~df["is_pit_in_lap"]) & (~df["is_pit_out_lap"]) & (~df["is_non_green"]) \
        & df["is_slick"] & df["has_lap_time"]
    if cfg.get("exclude_lap_one", True):
        base &= ~df["is_lap_one"]

    df["is_outlier"] = False
    for _, idx in df[base].groupby("session_key").groups.items():
        med = df.loc[idx, "lap_time_seconds"].median()
        df.loc[idx, "is_outlier"] = df.loc[idx, "lap_time_seconds"] > med * cfg["outlier_ratio"]

    df["is_clean_lap"] = base & ~df["is_outlier"]
    return df


def build_tyre_stints(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per (session, driver, stint)."""
    rows = []
    for (sk, drv, st), g in laps.dropna(subset=["stint_number"]).groupby(
            ["session_key", "driver_id", "stint_number"]):
        g = g.sort_values("lap_number")
        first = g.iloc[0]
        rows.append({
            "session_key": sk, "circuit_id": first["circuit_id"], "driver_id": drv,
            "stint_number": int(st), "compound": first["compound"],
            "stint_start_lap": int(g["lap_number"].min()), "stint_end_lap": int(g["lap_number"].max()),
            "stint_length": int(g["lap_number"].max() - g["lap_number"].min() + 1),
            # FastF1 TyreLife counts the current lap, so age at start = TyreLife(first lap) - 1.
            "tyre_age_at_start": float(first["tyre_life"] - 1) if pd.notna(first["tyre_life"]) else np.nan,
            "is_new_tyre": first["fresh_tyre"],
        })
    return pd.DataFrame(rows)


def _status_context(status: str) -> str:
    s = str(status)
    if "4" in s:
        return "FULL_SC"
    if "6" in s or "7" in s:
        return "VSC"
    if "5" in s:
        return "RED"
    return "GREEN"


def build_pit_stops(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per completed pit stop (in-lap k followed by out-lap k+1).

    Two time-loss estimates are computed, both relative to cars that stayed on track:

    * ``loss_vs_own_pace``: (in + out) - 2 * mean(driver's clean laps k-1 and k+2). Valid only
      when the surrounding laps are green, so it is NaN for neutralised stops.
    * ``loss_vs_field``: (in + out) - (median lap time on laps k and k+1 of drivers who did not
      pit on either lap). This reference shares the track status of the stop, so it is the
      estimator used for VSC / SC stops.

    ``pit_lane_duration_seconds`` = PitOutTime(k+1) - PitInTime(k): the time between the pit
    entry and exit timing points. It includes deceleration, limiter running and the stationary
    service, so it is NOT the stationary time. Stationary time is not in the public feed.
    """
    rows = []
    for sk, race in laps.groupby("session_key"):
        by_lap = race.set_index(["driver_id", "lap_number"])
        pitted_laps = set(zip(race.loc[race.is_pit_in_lap, "driver_id"], race.loc[race.is_pit_in_lap, "lap_number"])) | \
            set(zip(race.loc[race.is_pit_out_lap, "driver_id"], race.loc[race.is_pit_out_lap, "lap_number"]))
        for drv, d in race.groupby("driver_id"):
            d = d.set_index("lap_number").sort_index()
            stop_no = 0
            for k, row in d[d.is_pit_in_lap].iterrows():
                if (k + 1) not in d.index or not d.loc[k + 1, "is_pit_out_lap"]:
                    continue  # retirement in the pits or missing out-lap
                stop_no += 1
                out = d.loc[k + 1]
                in_t, out_t = row["lap_time_seconds"], out["lap_time_seconds"]

                def own_ref(lap):
                    if lap in d.index and bool(d.loc[lap, "is_clean_lap"]):
                        return d.loc[lap, "lap_time_seconds"]
                    return np.nan
                refs = [x for x in (own_ref(k - 1), own_ref(k + 2)) if pd.notna(x)]
                loss_own = (in_t + out_t - 2 * np.mean(refs)) if (refs and pd.notna(in_t) and pd.notna(out_t)) else np.nan

                field_ref = []
                for lap in (k, k + 1):
                    others = race[(race.lap_number == lap) & race.lap_time_seconds.notna()]
                    others = others[[(x, lap) not in pitted_laps for x in others.driver_id]]
                    field_ref.append(others["lap_time_seconds"].median() if len(others) >= 3 else np.nan)
                loss_field = (in_t + out_t - sum(field_ref)) if (pd.notna(in_t) and pd.notna(out_t) and all(pd.notna(field_ref))) else np.nan

                ctx = _status_context(str(row["track_status"]) + str(out["track_status"]))
                rows.append({
                    "session_key": sk, "circuit_id": row["circuit_id"], "driver_id": drv,
                    "pit_stop_number": stop_no, "pit_lap": int(k),
                    "pit_entry_session_s": row["pit_in_session_s"], "pit_exit_session_s": out["pit_out_session_s"],
                    "pit_lane_duration_seconds": out["pit_out_session_s"] - row["pit_in_session_s"],
                    "stationary_duration_seconds": np.nan,
                    "in_lap_seconds": in_t, "out_lap_seconds": out_t,
                    "loss_vs_own_pace": loss_own if ctx == "GREEN" else np.nan,
                    "loss_vs_field": loss_field,
                    "safety_car_context": ctx,
                    "compound_before": row["compound"], "compound_after": out["compound"],
                })
    return pd.DataFrame(rows)

