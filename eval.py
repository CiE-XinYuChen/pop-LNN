import os
import argparse
import csv

from typing import List, Dict

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import LTCNetwork


def mape(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Absolute Percentage Error in percentage points."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0


def evaluate(
    train_csv: str = "train.csv",
    model_dir: str = "models",
    plot_dir: str = "plots",
    result_csv: str = "mape_results.csv",
    hidden_size: int = 10,
):
    """Evaluate all country‑specific models with progressive forecasting.

    Steps for each country (LocID):
    1. Use 2009‑2018 (10 yrs) to predict 2019.
    2. Append predicted 2019 to 2010‑2018 actuals to predict 2020.
    3. Compute MAPE for 2019, 2020 and their average.
    4. Plot actual vs. predicted (2009‑2020) and save PNG.
    5. Collect per‑country errors into a CSV and print aggregate stats.
    """

    # ---------- I/O setup ----------
    os.makedirs(plot_dir, exist_ok=True)

    df = pd.read_csv(train_csv)
    device = torch.device("cpu")  # CPU evaluation is sufficient

    seq_years_1 = list(range(2009, 2019))  # 2009‑2018 inclusive
    seq_years_2 = list(range(2010, 2019))  # 2010‑2018 (+pred 2019 later)

    results: List[Dict[str, float]] = []

    for model_file in os.listdir(model_dir):
        if not model_file.endswith("_model.pth"):
            continue
        country_code = model_file.replace("_model.pth", "")
        model_path = os.path.join(model_dir, model_file)

        # ---- Extract this country's dataframe ----
        try:
            locid = int(country_code)
        except ValueError:
            print(f"⚠️  Skip {country_code}: name is not a numeric LocID")
            continue

        cdf = df[df["LocID"] == locid].sort_values("Time")
        required_years = set(seq_years_1 + [2019, 2020])
        if not required_years.issubset(set(cdf["Time"].values)):
            print(f"⚠️  Skip {country_code}: missing required years (2009‑2020)")
            continue

        pop_lookup = cdf.set_index("Time")["Population"].to_dict()

        # ---- Prepare first input window (2009‑2018) ----
        x1 = torch.tensor(
            [[[year, pop_lookup[year]] for year in seq_years_1]], dtype=torch.float32, device=device
        )  # shape (1, 10, 2)

        # ---- Load model ----
        model = LTCNetwork(input_size=2, hidden_size=hidden_size, output_size=1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            pred_2019, _ = model(x1)
            pred_2019 = pred_2019.item()

        # ---- Prepare second window (2010‑2018 + pred 2019) ----
        x2_features = [[year, pop_lookup[year]] for year in seq_years_2] + [[2019, pred_2019]]
        x2 = torch.tensor([x2_features], dtype=torch.float32, device=device)

        with torch.no_grad():
            pred_2020, _ = model(x2)
            pred_2020 = pred_2020.item()

        # ---- Compute errors ----
        actual_2019 = pop_lookup[2019]
        actual_2020 = pop_lookup[2020]
        mape_2019 = mape([actual_2019], [pred_2019])
        mape_2020 = mape([actual_2020], [pred_2020])
        mape_avg = mape([actual_2019, actual_2020], [pred_2019, pred_2020])

        results.append(
            {
                "LocID": locid,
                "MAPE_2019": mape_2019,
                "MAPE_2020": mape_2020,
                "MAPE_avg": mape_avg,
            }
        )

        # ---- Plot actual vs predicted ----
        years_plot = list(range(2009, 2021))  # 2009‑2020 inclusive
        actual_pop = [pop_lookup[y] for y in years_plot]
        predicted_pop = actual_pop.copy()
        predicted_pop[10] = pred_2019  # index 10 corresponds to 2019
        predicted_pop[11] = pred_2020  # index 11 corresponds to 2020

        plt.figure(figsize=(8, 5))
        plt.plot(years_plot, actual_pop, label="Actual", marker="o")
        plt.plot(years_plot, predicted_pop, label="Predicted", linestyle="--", marker="x")
        plt.title(f"Population Prediction — LocID {locid}")
        plt.xlabel("Year")
        plt.ylabel("Population")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{country_code}.png"))
        plt.close()

    # ---------- Save results ----------
    if not results:
        print("❌ No results generated. Did models and data align correctly?")
        return

    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["LocID", "MAPE_2019", "MAPE_2020", "MAPE_avg"])
        writer.writeheader()
        writer.writerows(results)

    # ---------- Aggregate statistics ----------
    mape_values = np.array([r["MAPE_avg"] for r in results])
    print("\n=== Aggregate MAPE statistics ===")
    print(f"Count of countries   : {len(mape_values)}")
    print(f"Mean MAPE (avg)      : {mape_values.mean():.2f}%")
    print(f"Std‑dev              : {mape_values.std(ddof=1):.2f}%")
    print(f"Min / Max            : {mape_values.min():.2f}% / {mape_values.max():.2f}%")
    print(f"Results CSV written  : {result_csv}")
    print(f"Plots folder         : {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LTCNetwork population models with progressive forecasting")
    parser.add_argument("--train_csv", default="train_merged.csv", help="Path to train.csv containing 2009‑2020 data")
    parser.add_argument("--model_dir", default="models", help="Directory with *_model.pth files")
    parser.add_argument("--plot_dir", default="plots", help="Where PNG plots are saved")
    parser.add_argument("--result_csv", default="mape_results.csv", help="CSV output for MAPE values")
    parser.add_argument("--hidden_size", type=int, default=10, help="Hidden size used in LTCNetwork (must match training)")

    args = parser.parse_args()
    evaluate(
        train_csv=args.train_csv,
        model_dir=args.model_dir,
        plot_dir=args.plot_dir,
        result_csv=args.result_csv,
        hidden_size=args.hidden_size,
    )
