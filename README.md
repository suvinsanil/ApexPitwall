# ApexPitwall 🏎️

**ML-Powered Formula 1 Pit-Stop Strategy Prediction & Simulation**

ApexPitwall is a Formula 1 race strategy engine built with Python, FastF1, LightGBM, and Streamlit. It models tyre degradation, isolates fuel burn-off, optimizes pit-stop strategy, and runs Monte Carlo race simulations.

---

## ⚡ How to Run the Project

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/suvinsanil/ApexPitwall.git
cd ApexPitwall

# Create and activate virtual environment
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard (Quick Start)
Pre-trained models and processed data are already included in the repo. You can run the dashboard immediately:
```bash
streamlit run app/streamlit_app.py
```
This opens the interactive app in your browser at `http://localhost:8501`, where you can:
- View telemetry pace and tyre wear degradation curves.
- Test 1-stop vs. 2-stop strategies with customizable pit laps.
- Run live Monte Carlo simulations (1,000 runs) to view win probabilities and confidence intervals.

---

### 3. Running the Backend Pipelines (Optional / From Scratch)
If you want to re-run the pipeline or retrain models from scratch:

```bash
# Step 1: Preprocess raw FastF1 race data (5 dry 2022 Grands Prix)
python src/data_pipeline.py

# Step 2: Train Ridge & LightGBM tyre degradation models
python src/train_model.py

# Step 3: Estimate empirical pit loss parameters from timing loops
python src/estimate_pit_loss.py

# Step 4: Run the 66-lap strategy simulation sweep (118 variants)
python src/strategy_simulator.py

# Step 5: Run unit tests
pytest tests
```

---

## 📊 Benchmark Results (2022 Spanish GP Held-Out Test)

| Model | MAE (seconds) | RMSE (seconds) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Naive Baseline (3-lap rolling pace)** | 0.5906 s | 0.8509 s | 0.6475 |
| **Ridge Regression Baseline** | 0.5520 s | 0.7512 s | 0.7265 |
| **LightGBM Regressor (Primary)** | **0.4633 s** | **0.6710 s** | **0.7779** |

- **Optimal Strategy:** On Catalunya's high-wear surface, a **2-stop Soft $\to$ Medium $\to$ Soft** (pits on laps 18, 40) is **32.8s to 44.3s faster** than 1-stop alternatives.
- **Measured Pit Loss:** Median green-flag pit loss at Catalunya is **22.59 seconds**.

---

## 📂 Project Structure

- `app/` — Streamlit interactive web application and view components.
- `src/` — Data cleaning, feature engineering, model training, and simulation engines (deterministic, counterfactual, Monte Carlo).
- `models/` — Saved LightGBM and Ridge model artifacts (`.joblib`).
- `data/processed/` — Processed lap telemetry, pit loss parameters, and strategy results.
- `reports/` — Empirical evaluation reports, markdown audits, and diagnostic figures.
- `tests/` — Pytest unit test suite verifying zero lookahead leakage and simulation logic.

---

## ⚖️ License
MIT License. Built for educational and research purposes using Formula 1 timing feeds via FastF1.
