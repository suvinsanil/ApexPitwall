"""Streamlit Telemetry & Lap Pace Component"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/race_laps_processed.csv"


@st.cache_data
def load_telemetry_data():
    df = pd.read_csv(DATA_PATH)
    return df


def render_telemetry_tab():
    st.header("🏁 Race Telemetry & Tyre Degradation Overview")
    df = load_telemetry_data()

    col1, col2 = st.columns([1, 3])
    with col1:
        circuits = df["Circuit"].unique().tolist()
        selected_circuit = st.selectbox("Select Circuit", circuits, index=circuits.index("catalunya") if "catalunya" in circuits else 0)

        circuit_df = df[df["Circuit"] == selected_circuit]
        drivers = sorted(circuit_df["Driver"].unique().tolist())
        selected_driver = st.selectbox("Select Driver", drivers, index=0)

    with col2:
        d_df = circuit_df[circuit_df["Driver"] == selected_driver].sort_values("LapNumber")

        # Telemetry plot
        fig, ax = plt.subplots(figsize=(10, 4.5))
        comp_palette = {"SOFT": "#E10600", "MEDIUM": "#D69E2E", "HARD": "#718096"}

        for comp in d_df["Compound"].unique():
            sub = d_df[d_df["Compound"] == comp]
            ax.scatter(sub["LapNumber"], sub["lap_time_seconds"], color=comp_palette.get(comp, "blue"), label=f"{comp} Stint", s=30, alpha=0.8)

        ax.plot(d_df["LapNumber"], d_df["rolling_pace_mean_3"], color="black", linestyle="--", linewidth=1.5, label="3-Lap Rolling Pace (Shifted)")
        ax.set_title(f"{selected_driver} — Lap Time Pace Profile ({selected_circuit.capitalize()})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Lap Number", fontsize=10)
        ax.set_ylabel("Lap Time (seconds)", fontsize=10)
        ax.legend(frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Tyre Degradation Slopes by Compound")
    c_df = df[df["Circuit"] == selected_circuit]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        comp_data = c_df[c_df["Compound"] == comp]
        if len(comp_data) > 0:
            ax2.scatter(comp_data["TyreLife"], comp_data["lap_time_seconds"], alpha=0.2, color=comp_palette.get(comp, "gray"), s=20)
            if len(comp_data) >= 15:
                p = np.polyfit(comp_data["TyreLife"], comp_data["lap_time_seconds"], 1)
                x_vals = np.linspace(comp_data["TyreLife"].min(), comp_data["TyreLife"].max(), 30)
                ax2.plot(x_vals, np.poly1d(p)(x_vals), color=comp_palette.get(comp, "black"), linewidth=2, label=f"{comp} Wear ({p[0]:+.3f} s/lap)")

    ax2.set_title(f"Empirical Tyre Degradation vs. Tyre Age ({selected_circuit.capitalize()})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Tyre Age (Laps)", fontsize=10)
    ax2.set_ylabel("Lap Time (seconds)", fontsize=10)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig2)

