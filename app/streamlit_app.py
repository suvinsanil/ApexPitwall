"""ApexPitwall — Interactive Streamlit Dashboard"""

import streamlit as st
import pandas as pd
import numpy as np

from app.components.telemetry_view import render_telemetry_tab
from app.components.strategy_view import render_strategy_tab

st.set_page_config(
    page_title="ApexPitwall | F1 Strategy Simulator",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ ApexPitwall: Formula 1 Strategy Prediction & Simulation")
st.caption("ML-Powered Pit Stop Optimization, Degradation Analysis, and Monte Carlo Uncertainty Quantification")

tab1, tab2, tab3 = st.tabs([
    "📊 Race Telemetry & Degradation",
    "⚡ Strategy Simulator & Monte Carlo",
    "📖 Model Architecture & Evaluation"
])

with tab1:
    render_telemetry_tab()

with tab2:
    render_strategy_tab()

with tab3:
    st.header("🔬 Model Architecture & Evaluation Benchmark")
    st.markdown("""
    ### Primary Evaluation Summary (2022 Spanish Grand Prix Held-Out Test)
    All models were trained on 4 championship rounds (Bahrain, Zandvoort, Mexico City, Abu Dhabi)
    and strictly evaluated out-of-sample on the held-out **2022 Spanish Grand Prix (Catalunya)**.

    | Model Architecture | Calibration Strategy | Held-Out MAE (s) | Held-Out RMSE (s) | $R^2$ Score |
    | :--- | :--- | :---: | :---: | :---: |
    | **Naive Baseline** | 3-Lap Shifted Rolling Mean ($t-1$) | **0.5906 s** | **0.8509 s** | 0.6475 |
    | **Ridge Regression** | Cross-Circuit Held-Out | **1.0189 s** | **1.2070 s** | 0.2907 |
    | **LightGBM Regressor** | Cross-Circuit Held-Out | **3.0252 s** | **3.1822 s** | -3.9305 |
    | **Ridge Regression** | Within-Circuit 5-Fold Cross Validation | **0.5520 s** | **0.7512 s** | 0.7265 |
    | **LightGBM Regressor** | Within-Circuit 5-Fold Cross Validation | **0.4633 s** | **0.6710 s** | **0.7779** |

    ### Core Architectural Takeaways
    1. **Leakage Prevention:** Features strictly apply a `.shift(1)` step before computing rolling pace metrics.
    2. **Physics Fuel Disentanglement:** Fuel mass decreases linearly ($110 \\times (1 - \\text{lap}/N)$), isolating true tyre degradation (+0.04 to +0.06 s/lap) from vehicle lightening.
    3. **Strategy Optimization Verdict:** On Catalunya's high-wear surface, a **2-stop Soft -> Medium -> Soft strategy (stopping on Laps 18, 40)** beats 1-stop alternatives by **32 to 44 seconds**.
    """)
