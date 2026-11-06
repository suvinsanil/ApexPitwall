"""Tests for the strategy simulation engine."""

import numpy as np
import pytest

from src.strategy_simulator import simulate_strategy_rollout, load_models, load_pit_parameters


def test_strategy_simulator_rollout():
    lgbm_model, ridge_model = load_models()
    pit_params = load_pit_parameters()
    pit_loss = pit_params.get("recommended_simulation_pit_loss_seconds", 22.586)

    # 1-stop simulation: Soft -> Medium, pit at lap 20
    res_1stop = simulate_strategy_rollout(
        compounds=["SOFT", "MEDIUM"],
        pit_laps=[20],
        model=ridge_model,
        model_type="ridge",
        pit_loss_sec=pit_loss,
        total_laps=66
    )

    assert len(res_1stop["lap_times"]) == 66
    assert len(res_1stop["stints"]) == 2
    assert res_1stop["stints"][0]["laps"] == 20
    assert res_1stop["stints"][1]["laps"] == 46
    assert np.isclose(res_1stop["pit_penalty_s"], pit_loss)
    assert np.isclose(res_1stop["total_race_time_s"], res_1stop["driving_time_s"] + pit_loss)

    # 2-stop simulation: Soft -> Medium -> Medium, pits at 18 and 38
    res_2stop = simulate_strategy_rollout(
        compounds=["SOFT", "MEDIUM", "MEDIUM"],
        pit_laps=[18, 38],
        model=ridge_model,
        model_type="ridge",
        pit_loss_sec=pit_loss,
        total_laps=66
    )
    assert len(res_2stop["stints"]) == 3
    assert np.isclose(res_2stop["pit_penalty_s"], 2 * pit_loss)

