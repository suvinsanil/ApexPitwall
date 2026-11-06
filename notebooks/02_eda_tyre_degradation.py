"""
ApexPitwall — Exploratory Data Analysis: Tyre Degradation
Path: notebooks/02_eda_tyre_degradation.py

Performs empirical exploratory analysis on lap time vs tyre age across compounds
and circuits, evaluates confounders (fuel mass, track temperature), and produces
publication-quality figures in reports/figures/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300

FIGURES_DIR = "reports/figures"
DATA_PATH = "data/processed/race_laps_processed.csv"

# Compound colors matching standard F1 Pirelli branding
COMPOUND_COLORS = {
    "SOFT": "#E10600",      # Red
    "MEDIUM": "#FFF200",    # Yellow
    "HARD": "#FFFFFF",      # White (or light slate gray for visibility)
}
COMPOUND_EDGE_COLORS = {
    "SOFT": "#8B0000",
    "MEDIUM": "#B8860B",
    "HARD": "#4A5568",
}


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Loads and inspects the processed race laps dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found at {path}. Run src/data_pipeline.py first.")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} laps from {path}")
    return df


def plot_lap_time_vs_tyre_age(df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """
    Plots lap time vs. tyre age by compound for Catalunya (Spain R06)
    and Bahrain (R01), illustrating empirical tyre wear curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    circuits = [
        ("catalunya", "Spanish Grand Prix (Catalunya - High Lateral Energy)", axes[0]),
        ("bahrain", "Bahrain Grand Prix (Sakhir - High Longitudinal Traction)", axes[1]),
    ]

    for circuit_id, title, ax in circuits:
        c_df = df[df["Circuit"] == circuit_id]

        for compound in ["SOFT", "MEDIUM", "HARD"]:
            comp_data = c_df[c_df["Compound"] == compound]
            if len(comp_data) == 0:
                continue

            # Scatter points with alpha
            ax.scatter(
                comp_data["TyreLife"],
                comp_data["lap_time_seconds"],
                c=COMPOUND_COLORS.get(compound, "gray"),
                edgecolors=COMPOUND_EDGE_COLORS.get(compound, "black"),
                alpha=0.35,
                s=28,
                label=f"{compound} (n={len(comp_data)})"
            )

            # Polynomial trendline (degree 1 or 2)
            if len(comp_data) >= 10:
                z = np.polyfit(comp_data["TyreLife"], comp_data["lap_time_seconds"], 1)
                p = np.poly1d(z)
                x_vals = np.linspace(comp_data["TyreLife"].min(), comp_data["TyreLife"].max(), 50)
                slope_per_lap = z[0]
                ax.plot(
                    x_vals, p(x_vals),
                    color=COMPOUND_EDGE_COLORS.get(compound, "black"),
                    linestyle="--",
                    linewidth=2.2,
                    label=f"{compound} Trend ({slope_per_lap:+.3f} s/lap)"
                )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Tyre Age (Laps on Set)", fontsize=11)
        ax.set_ylabel("Lap Time (s)", fontsize=11)
        ax.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Empirical Tyre Degradation: Lap Time vs. Tyre Age", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = os.path.join(output_dir, "lap_time_vs_tyre_age.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_compound_pace_comparison(df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """
    Plots lap time distributions across circuits and compounds (box plots and violin plots).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))

    circuit_order = ["zandvoort", "rodriguez", "catalunya", "yas_marina", "bahrain"]
    compound_order = ["SOFT", "MEDIUM", "HARD"]

    palette = {"SOFT": "#E10600", "MEDIUM": "#E5A910", "HARD": "#6C757D"}

    sns.boxplot(
        data=df,
        x="Circuit",
        y="lap_time_seconds",
        hue="Compound",
        order=circuit_order,
        hue_order=compound_order,
        palette=palette,
        ax=ax,
        fliersize=2,
        linewidth=1.2,
        boxprops=dict(alpha=0.85)
    )

    ax.set_title("Lap Time Distribution by Circuit and Compound (2022 Season Clean Laps)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Circuit", fontsize=11)
    ax.set_ylabel("Lap Time (s)", fontsize=11)
    ax.legend(title="Compound", frameon=True, facecolor="#F8F9FA")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_file = os.path.join(output_dir, "compound_pace_comparison.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_track_temp_vs_pace(df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """
    Analyzes track temperature vs pace and fuel mass confounding effect.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Track Temp vs Lap Time across Catalunya
    cat_df = df[df["Circuit"] == "catalunya"].copy()
    scatter1 = axes[0].scatter(
        cat_df["TrackTemp"],
        cat_df["lap_time_seconds"],
        c=cat_df["fuel_remaining_kg"],
        cmap="viridis_r",
        s=30,
        alpha=0.6,
        edgecolors="none"
    )
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label("Fuel Mass Remaining (kg)", fontsize=10)
    axes[0].set_title("Catalunya: Track Temp vs Lap Time (Color: Fuel Mass)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Track Temperature (°C)", fontsize=10)
    axes[0].set_ylabel("Lap Time (s)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Right: Fuel Mass vs Lap Time across Catalunya
    scatter2 = axes[1].scatter(
        cat_df["fuel_remaining_kg"],
        cat_df["lap_time_seconds"],
        c=cat_df["TyreLife"],
        cmap="plasma",
        s=30,
        alpha=0.6,
        edgecolors="none"
    )
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label("Tyre Life (Laps)", fontsize=10)
    axes[1].set_title("Catalunya: Fuel Burn Effect vs Lap Time (Color: Tyre Age)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Fuel Remaining (kg)", fontsize=10)
    axes[1].set_ylabel("Lap Time (s)", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Confounder Analysis: Fuel Mass vs. Track Temperature vs. Tyre Age", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = os.path.join(output_dir, "track_temp_vs_pace.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def print_eda_statistics(df: pd.DataFrame):
    """Prints empirical statistical summary to stdout."""
    print("\n" + "=" * 80)
    print("EMPIRICAL EDA SUMMARY: SAMPLE COUNTS & COVERAGE")
    print("=" * 80)
    c_summary = df.groupby(["Circuit", "Compound"]).agg(
        LapCount=("lap_time_seconds", "count"),
        MeanLapTime=("lap_time_seconds", "mean"),
        StdLapTime=("lap_time_seconds", "std"),
        MinTyreLife=("TyreLife", "min"),
        MaxTyreLife=("TyreLife", "max"),
        MeanTrackTemp=("TrackTemp", "mean"),
    ).round(3)
    print(c_summary.to_string())

    print("\n" + "=" * 80)
    print("LINEAR DEGRADATION FIT (RAW SLOPE IN SECONDS PER LAP OF TYRE AGE)")
    print("=" * 80)
    slopes = []
    for (circuit, compound), group in df.groupby(["Circuit", "Compound"]):
        if len(group) >= 15:
            poly = np.polyfit(group["TyreLife"], group["lap_time_seconds"], 1)
            slopes.append({
                "Circuit": circuit,
                "Compound": compound,
                "Laps": len(group),
                "Slope_s_per_lap": round(poly[0], 4),
                "BasePace_s": round(poly[1], 3)
            })
    slope_df = pd.DataFrame(slopes)
    print(slope_df.to_string(index=False))


def run_eda():
    df = load_dataset()
    print_eda_statistics(df)
    plot_lap_time_vs_tyre_age(df)
    plot_compound_pace_comparison(df)
    plot_track_temp_vs_pace(df)
    print("\nEDA Completed successfully.")


if __name__ == "__main__":
    run_eda()

