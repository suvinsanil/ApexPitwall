"""Stage D: Counterfactual Analysis Engine

Evaluates perturbations to historical driver race strategies while holding other factors constant:
1. Pit Window Shift (pitting k laps earlier/later)
2. Compound Substitution (fitting Hard vs Medium vs Soft)
3. Stint Count Switch (1-stop vs 2-stop conversion)
4. Virtual Safety Car (VSC) exploitation vs Green Flag pit stop
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.strategy_simulator import simulate_strategy_rollout, load_models, load_pit_parameters


class CounterfactualEngine:
    def __init__(self, model_type: str = "lgbm"):
        self.lgbm_model, self.ridge_model = load_models()
        self.pit_params = load_pit_parameters()
        self.default_pit_loss = self.pit_params.get("recommended_simulation_pit_loss_seconds", 22.586)
        self.model_type = model_type
        self.model = self.lgbm_model if model_type == "lgbm" else self.ridge_model

    def evaluate_scenario(
        self,
        baseline_compounds: List[str],
        baseline_pit_laps: List[int],
        counterfactual_compounds: List[str],
        counterfactual_pit_laps: List[int],
        baseline_pit_loss: float | None = None,
        counterfactual_pit_loss: float | None = None,
        total_laps: int = 66
    ) -> Dict[str, Any]:
        """
        Calculates the counterfactual time delta:
        Delta_T = Total_Time(Baseline) - Total_Time(Counterfactual)
        Positive Delta_T indicates counterfactual is FASTER.
        """
        p_loss_base = baseline_pit_loss if baseline_pit_loss is not None else self.default_pit_loss
        p_loss_cf = counterfactual_pit_loss if counterfactual_pit_loss is not None else self.default_pit_loss

        sim_base = simulate_strategy_rollout(
            compounds=baseline_compounds,
            pit_laps=baseline_pit_laps,
            model=self.model,
            model_type=self.model_type,
            pit_loss_sec=p_loss_base,
            total_laps=total_laps
        )

        sim_cf = simulate_strategy_rollout(
            compounds=counterfactual_compounds,
            pit_laps=counterfactual_pit_laps,
            model=self.model,
            model_type=self.model_type,
            pit_loss_sec=p_loss_cf,
            total_laps=total_laps
        )

        delta_t_s = sim_base["total_race_time_s"] - sim_cf["total_race_time_s"]

        return {
            "delta_t_seconds": round(delta_t_s, 2),
            "baseline_total_time_s": round(sim_base["total_race_time_s"], 2),
            "counterfactual_total_time_s": round(sim_cf["total_race_time_s"], 2),
            "baseline_driving_time_s": round(sim_base["driving_time_s"], 2),
            "counterfactual_driving_time_s": round(sim_cf["driving_time_s"], 2),
            "baseline_pit_penalty_s": round(sim_base["pit_penalty_s"], 2),
            "counterfactual_pit_penalty_s": round(sim_cf["pit_penalty_s"], 2),
            "baseline_stints": sim_base["stints"],
            "counterfactual_stints": sim_cf["stints"],
            "baseline_lap_times": sim_base["lap_times"],
            "counterfactual_lap_times": sim_cf["lap_times"],
            "is_faster": delta_t_s > 0
        }

