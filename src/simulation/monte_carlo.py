"""Stage E: Vectorized Monte Carlo Uncertainty Simulator

Simulates stochastic distributions of race outcomes incorporating:
1. Base Driver Pace Noise (autoregressive AR(1) process)
2. Stochastic Pit Stop Durations (Gamma-distributed stationary duration)
3. Tyre degradation variance
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List

from src.strategy_simulator import simulate_strategy_rollout, load_models, load_pit_parameters


class MonteCarloSimulator:
    def __init__(self, model_type: str = "ridge"):
        self.lgbm_model, self.ridge_model = load_models()
        self.pit_params = load_pit_parameters()
        self.model_type = model_type
        self.model = self.ridge_model if model_type == "ridge" else self.lgbm_model
        self.base_pit_loss = self.pit_params.get("recommended_simulation_pit_loss_seconds", 22.586)

    def simulate_strategy_distributions(
        self,
        strategy_1: Dict[str, Any],
        strategy_2: Dict[str, Any],
        n_sims: int = 1000,
        sigma_pace: float = 0.25,
        rho_ar1: float = 0.4,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Runs n_sims stochastic trials for two candidate strategies.
        Each strategy dict must contain: 'name', 'compounds', 'pit_laps'.
        """
        np.random.seed(random_seed)

        # 1. Deterministic base trajectories
        det_1 = simulate_strategy_rollout(
            compounds=strategy_1["compounds"],
            pit_laps=strategy_1["pit_laps"],
            model=self.model,
            model_type=self.model_type,
            pit_loss_sec=self.base_pit_loss,
            total_laps=66
        )

        det_2 = simulate_strategy_rollout(
            compounds=strategy_2["compounds"],
            pit_laps=strategy_2["pit_laps"],
            model=self.model,
            model_type=self.model_type,
            pit_loss_sec=self.base_pit_loss,
            total_laps=66
        )

        base_laps_1 = np.array(det_1["lap_times"])
        base_laps_2 = np.array(det_2["lap_times"])
        num_stops_1 = len(strategy_1["pit_laps"])
        num_stops_2 = len(strategy_2["pit_laps"])

        # 2. Generate AR(1) pace noise matrix: (n_sims, 66)
        def generate_ar1_noise():
            noise = np.zeros((n_sims, 66))
            eps = np.random.normal(0, sigma_pace * np.sqrt(1 - rho_ar1**2), size=(n_sims, 66))
            noise[:, 0] = np.random.normal(0, sigma_pace, size=n_sims)
            for t in range(1, 66):
                noise[:, t] = rho_ar1 * noise[:, t - 1] + eps[:, t]
            return noise

        noise_1 = generate_ar1_noise()
        noise_2 = generate_ar1_noise()

        # 3. Pit stop duration noise: base transit (~19.8s) + Gamma stationary distribution
        # Gamma(shape=25, scale=0.1) has mean 2.5s, variance 0.25s
        def sample_pit_penalties(num_stops):
            if num_stops == 0:
                return np.zeros(n_sims)
            # Sample for each stop and sum
            stops_total = np.zeros(n_sims)
            for _ in range(num_stops):
                transit = np.random.normal(19.8, 0.4, size=n_sims)
                # Add stationary stop time (Gamma distributed with occasional long tail)
                stationary = np.random.gamma(shape=25.0, scale=0.11, size=n_sims)
                stops_total += (transit + stationary)
            return stops_total

        pit_penalties_1 = sample_pit_penalties(num_stops_1)
        pit_penalties_2 = sample_pit_penalties(num_stops_2)

        # 4. Total simulated race times
        times_1 = np.sum(base_laps_1 + noise_1, axis=1) + pit_penalties_1
        times_2 = np.sum(base_laps_2 + noise_2, axis=1) + pit_penalties_2

        # Probability strategy 1 beats strategy 2
        prob_1_beats_2 = float(np.mean(times_1 < times_2))

        return {
            "n_sims": n_sims,
            "strategy_1_name": strategy_1["name"],
            "strategy_2_name": strategy_2["name"],
            "prob_1_beats_2": round(prob_1_beats_2, 4),
            "prob_2_beats_1": round(1.0 - prob_1_beats_2, 4),
            "strat_1_mean_time": round(float(np.mean(times_1)), 2),
            "strat_2_mean_time": round(float(np.mean(times_2)), 2),
            "strat_1_ci90": [round(float(np.percentile(times_1, 5)), 2), round(float(np.percentile(times_1, 95)), 2)],
            "strat_2_ci90": [round(float(np.percentile(times_2, 5)), 2), round(float(np.percentile(times_2, 95)), 2)],
            "strat_1_distribution": times_1,
            "strat_2_distribution": times_2
        }
