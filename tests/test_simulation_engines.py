"""Tests for Counterfactual Analysis and Monte Carlo uncertainty quantification."""

import numpy as np
import pytest

from src.simulation.counterfactual import CounterfactualEngine
from src.simulation.monte_carlo import MonteCarloSimulator


def test_counterfactual_sanity_zero_delta():
    """Identical counterfactual must return exactly 0.00s delta."""
    engine = CounterfactualEngine(model_type="ridge")
    res = engine.evaluate_scenario(
        baseline_compounds=["SOFT", "MEDIUM"],
        baseline_pit_laps=[20],
        counterfactual_compounds=["SOFT", "MEDIUM"],
        counterfactual_pit_laps=[20]
    )
    assert np.isclose(res["delta_t_seconds"], 0.0, atol=0.01)


def test_counterfactual_vsc_advantage():
    """Pitting under VSC (saving ~10s in pit loss) must show positive time delta."""
    engine = CounterfactualEngine(model_type="ridge")
    res = engine.evaluate_scenario(
        baseline_compounds=["SOFT", "MEDIUM"],
        baseline_pit_laps=[22],
        counterfactual_compounds=["SOFT", "MEDIUM"],
        counterfactual_pit_laps=[22],
        baseline_pit_loss=22.586,
        counterfactual_pit_loss=12.0
    )
    # Counterfactual should save approximately 10.58s
    assert res["is_faster"]
    assert np.isclose(res["delta_t_seconds"], 10.59, atol=0.1)


def test_monte_carlo_distribution():
    mc = MonteCarloSimulator(model_type="ridge")
    s1 = {"name": "2-Stop Soft-Med-Soft", "compounds": ["SOFT", "MEDIUM", "SOFT"], "pit_laps": [18, 40]}
    s2 = {"name": "1-Stop Soft-Hard", "compounds": ["SOFT", "HARD"], "pit_laps": [28]}

    res = mc.simulate_strategy_distributions(s1, s2, n_sims=100)
    assert res["n_sims"] == 100
    assert 0.0 <= res["prob_1_beats_2"] <= 1.0
    # 2-stop should have a higher win probability on Catalunya
    assert res["prob_1_beats_2"] > 0.60
