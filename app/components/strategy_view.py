"""Streamlit Strategy View Component"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.strategy_simulator import simulate_strategy_rollout, load_models, load_pit_parameters
from src.simulation.counterfactual import CounterfactualEngine
from src.simulation.monte_carlo import MonteCarloSimulator


def render_strategy_tab():
    st.header("⚡ Race Strategy Simulator & Counterfactual Engine")
    st.markdown("""
    Evaluate multi-stint pit strategies, explore interactive 'what-if' counterfactual perturbations,
    and simulate Monte Carlo race outcome distributions.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Strategy Parameters")
        strat_mode = st.radio("Strategy Type", ["1-Stop", "2-Stop"], index=1)
        model_choice = st.selectbox("Inference Model", ["LightGBM (Primary)", "Ridge (Linear Sensitivity)"])
        model_type = "lgbm" if "LightGBM" in model_choice else "ridge"

        pit_loss_val = st.slider("Pit Loss Penalty (s)", 15.0, 30.0, 22.59, 0.1)

        if strat_mode == "1-Stop":
            c1 = st.selectbox("Stint 1 Compound", ["SOFT", "MEDIUM", "HARD"], index=0)
            c2 = st.selectbox("Stint 2 Compound", ["SOFT", "MEDIUM", "HARD"], index=1)
            p1 = st.slider("Pit Stop Lap", 10, 50, 24)
            compounds = [c1, c2]
            pit_laps = [p1]
        else:
            c1 = st.selectbox("Stint 1 Compound", ["SOFT", "MEDIUM", "HARD"], index=0)
            c2 = st.selectbox("Stint 2 Compound", ["SOFT", "MEDIUM", "HARD"], index=1)
            c3 = st.selectbox("Stint 3 Compound", ["SOFT", "MEDIUM", "HARD"], index=0)
            p1 = st.slider("Pit Stop 1 Lap", 10, 30, 18)
            p2 = st.slider("Pit Stop 2 Lap", p1 + 8, 55, 40)
            compounds = [c1, c2, c3]
            pit_laps = [p1, p2]

    # Run simulation
    lgbm_model, ridge_model = load_models()
    active_model = lgbm_model if model_type == "lgbm" else ridge_model

    sim_res = simulate_strategy_rollout(
        compounds=compounds,
        pit_laps=pit_laps,
        model=active_model,
        model_type=model_type,
        pit_loss_sec=pit_loss_val,
        total_laps=66
    )

    with col2:
        st.subheader("Simulated Race Performance")
        tot_time = sim_res["total_race_time_s"]
        mins = int(tot_time // 60)
        secs = tot_time % 60

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Race Time", f"{mins}m {secs:05.2f}s")
        m2.metric("Net Driving Time", f"{sim_res['driving_time_s']:.2f} s")
        m3.metric("Total Pit Loss", f"{sim_res['pit_penalty_s']:.2f} s")

        # Stint breakdown table
        stint_df = pd.DataFrame(sim_res["stints"])[["stint", "compound", "start_lap", "end_lap", "laps", "mean_pace_s"]]
        stint_df["mean_pace_s"] = stint_df["mean_pace_s"].round(3)
        st.dataframe(stint_df, use_container_width=True)

        # Stint Gantt Chart
        fig, ax = plt.subplots(figsize=(8, 1.8))
        comp_palette = {"SOFT": "#E10600", "MEDIUM": "#D69E2E", "HARD": "#718096"}
        left = 0
        for s in sim_res["stints"]:
            ax.barh(0, s["laps"], left=left, color=comp_palette.get(s["compound"], "gray"), edgecolor="black", height=0.6)
            ax.text(left + s["laps"]/2, 0, f"{s['compound'][0]} ({s['laps']}L)", ha="center", va="center", color="white", fontweight="bold")
            left += s["laps"]
        ax.set_xlim(0, 66)
        ax.set_yticks([])
        ax.set_xlabel("Lap Number (Race Distance: 66 Laps)")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🎲 Monte Carlo Uncertainty Comparison")
    st.caption("Compare your selected strategy against the optimal benchmark (Soft -> Medium -> Soft on Laps 18, 40).")

    if st.button("Run 1,000 Monte Carlo Simulations"):
        with st.spinner("Simulating stochastic driver pace and pit stops..."):
            mc = MonteCarloSimulator(model_type="ridge")
            s_custom = {"name": "User Strategy", "compounds": compounds, "pit_laps": pit_laps}
            s_bench = {"name": "Benchmark (2-Stop S-M-S)", "compounds": ["SOFT", "MEDIUM", "SOFT"], "pit_laps": [18, 40]}
            mc_res = mc.simulate_strategy_distributions(s_custom, s_bench, n_sims=1000)

            c_mc1, c_mc2 = st.columns([1, 2])
            with c_mc1:
                st.metric("Win Probability vs Benchmark", f"{mc_res['prob_1_beats_2'] * 100:.1f} %")
                st.write(f"**User 90% CI:** {mc_res['strat_1_ci90'][0]}s – {mc_res['strat_1_ci90'][1]}s")
                st.write(f"**Benchmark 90% CI:** {mc_res['strat_2_ci90'][0]}s – {mc_res['strat_2_ci90'][1]}s")

            with c_mc2:
                fig_mc, ax_mc = plt.subplots(figsize=(8, 3.5))
                ax_mc.hist(mc_res["strat_1_distribution"], bins=30, alpha=0.5, color="#2B6CB0", label="User Strategy")
                ax_mc.hist(mc_res["strat_2_distribution"], bins=30, alpha=0.5, color="#C53030", label="Benchmark (S-M-S)")
                ax_mc.set_xlabel("Total Simulated Race Time (seconds)")
                ax_mc.set_ylabel("Frequency")
                ax_mc.legend()
                ax_mc.grid(True, linestyle=":", alpha=0.6)
                st.pyplot(fig_mc)
