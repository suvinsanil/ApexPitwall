# ApexPitwall 🏎️

**ML-Powered Formula 1 Tyre Degradation Prediction & Pit-Stop Strategy Optimization**

ApexPitwall is a Formula 1 race strategy engine built using Python, FastF1, LightGBM, and Streamlit. It uses historical race telemetry to model tyre degradation, account for fuel burn-off, estimate pit-stop time loss, and evaluate alternative race strategies through deterministic and Monte Carlo simulations.

## Features

- **Tyre Degradation Prediction:** Predicts lap-time evolution using machine learning models trained on historical F1 telemetry.
- **Fuel Burn-off Adjustment:** Separates fuel-related lap-time improvements from tyre degradation.
- **Pit-Loss Estimation:** Estimates empirical pit-stop time loss using historical timing data.
- **Strategy Optimization:** Evaluates different tyre compounds, stint lengths, and pit-stop combinations.
- **Monte Carlo Simulation:** Models race-time uncertainty and compares strategy outcomes across simulation runs.
- **Interactive Dashboard:** Visualizes telemetry, tyre degradation, strategy comparisons, and simulation results through Streamlit.

---

## How to Run the Project

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/suvinsanil/ApexPitwall.git
cd ApexPitwall

# Create a virtual environment
python -m venv .venv

# Activate it

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Dashboard

Pre-trained models and processed datasets are included in the repository.

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

The dashboard allows you to:

- Explore lap-time telemetry and tyre degradation curves.
- Compare one-stop and two-stop strategies.
- Customize tyre compounds and pit-stop laps.
- Run Monte Carlo simulations to examine race-time distributions and strategy probabilities.

---

## Running the Backend Pipelines

The following scripts reproduce the data-processing, model-training, and strategy-evaluation workflow.

```bash
# Step 1: Process historical FastF1 telemetry
python src/data_pipeline.py

# Step 2: Train Ridge Regression and LightGBM models
python src/train_model.py

# Step 3: Estimate empirical pit-stop time loss
python src/estimate_pit_loss.py

# Step 4: Evaluate candidate race strategies
python src/strategy_simulator.py

# Step 5: Run unit tests
pytest tests
```

### Pipeline Overview

1. **Data Processing:** Collects and cleans dry-weather telemetry from five 2022 Grands Prix.
2. **Model Training:** Trains Ridge Regression and LightGBM models using four circuits.
3. **Model Evaluation:** Evaluates generalization performance on the held-out Catalunya circuit.
4. **Pit-Loss Estimation:** Calculates empirical green-flag pit-stop time loss.
5. **Strategy Simulation:** Sweeps 118 strategy configurations for a 66-lap Catalunya race.

---

## Benchmark Results

### 2022 Spanish Grand Prix — Held-out Circuit Evaluation

| Model | MAE (seconds) | RMSE (seconds) | R² Score |
|---|---:|---:|---:|
| Naive Baseline (3-lap rolling pace) | 0.5906 | 0.8509 | 0.6475 |
| Ridge Regression | 0.5520 | 0.7512 | 0.7265 |
| **LightGBM Regressor** | **0.4633** | **0.6710** | **0.7779** |

**Evaluation methodology:** Models were trained on Bahrain, Zandvoort, Mexico, and Abu Dhabi telemetry, with Catalunya reserved as the held-out test circuit.

### Strategy Optimization Results

| Parameter | Result |
|---|---|
| Circuit | Circuit de Barcelona-Catalunya |
| Race Distance | 66 laps |
| Strategy Configurations Evaluated | 118 |
| Best Model | LightGBM |
| Optimal Strategy | **Soft → Medium → Soft** |
| Pit Stops | 2 |
| Pit Laps | 18, 40 |
| Stint Lengths | 18, 22, 26 laps |
| Predicted Race Time | **5902.43 seconds** |
| Improvement Over Next-best Strategy | **4.01 seconds** |
| Estimated Green-flag Pit Loss | **22.59 seconds** |

*Strategy results represent model-based predictions under the simulator's assumptions, not a guarantee of real-world race outcomes.*

---

## Project Structure

```text
ApexPitwall/
│
├── app/
│   ├── components/
│   └── streamlit_app.py
│
├── src/
│   ├── data_pipeline.py
│   ├── train_model.py
│   ├── estimate_pit_loss.py
│   └── strategy_simulator.py
│
├── models/
│   └── Trained model artifacts (.joblib)
│
├── data/
│   └── processed/
│
├── reports/
│   └── Evaluation reports and figures
│
├── tests/
│
├── requirements.txt
└── README.md
```

## Testing

The project includes automated tests covering:

- Data loading and processing.
- Feature engineering and lookahead leakage prevention.
- Strategy simulation logic.
- Simulation engine behavior.

Run the test suite:

```bash
pytest tests
```

**Latest test result:** 7 tests passed.

---

## Tech Stack

- **Language:** Python
- **Machine Learning:** LightGBM, Scikit-learn, Ridge Regression
- **Data Processing:** Pandas, NumPy
- **F1 Telemetry:** FastF1
- **Visualization:** Matplotlib, Streamlit
- **Testing:** Pytest

## License

MIT License. Built for educational and research purposes using Formula 1 timing data accessed through FastF1.