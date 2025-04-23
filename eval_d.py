#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate country-specific LTCNetwork models with progressive forecasting.
author: your-name
"""

import os, csv, argparse, warnings, math
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model import LTCNetwork
warnings.filterwarnings("ignore", category=UserWarning)  # matplotlib font

# ---------- 误差指标 ----------
def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def medape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

# ---------- 主流程 ----------
def evaluate(train_csv="train.csv",
             model_dir="models",
             plot_dir="plots",
             result_csv="mape_results.csv",
             hidden_size=10,
             device="cpu"):

    os.makedirs(plot_dir, exist_ok=True)
    df = pd.read_csv(train_csv)

    seq1 = list(range(2009, 2019))       # 2009-2018
    seq2 = list(range(2010, 2019))       # 2010-2018
    need_years = set(seq1 + [2019, 2020])

    per_country: List[Dict] = []

    for file in os.listdir(model_dir):
        if not file.endswith("_model.pth"):
            continue
        code = file.replace("_model.pth", "")
        try:
            locid = int(code)
        except ValueError:
            print(f"Skip {file}: 非数字 LocID")
            continue

        cdf = df[df["LocID"] == locid].sort_values("Time")
        if not need_years.issubset(set(cdf["Time"])):
            print(f"Skip {locid}: 缺少 2009-2020 完整数据")
            continue

        pop = cdf.set_index("Time")["Population"].to_dict()

        # -------- 载入模型 --------
        model = LTCNetwork(input_size=2, hidden_size=hidden_size, output_size=1).to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, file), map_location=device))
        model.eval()

        # -------- 第一次窗口 (→2019) --------
        x1 = torch.tensor([[[y, pop[y]] for y in seq1]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred19 = float(model(x1)[0])

        # -------- 第二次窗口 (→2020) --------
        x2_feat = [[y, pop[y]] for y in seq2] + [[2019, pred19]]
        x2 = torch.tensor([x2_feat], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred20 = float(model(x2)[0])

        # -------- 误差 --------
        true19, true20 = pop[2019], pop[2020]
        metrics = dict(
            LocID=locid,
            MAPE_19=mape([true19], [pred19]),
            MAPE_20=mape([true20], [pred20]),
            MAE_19=abs(pred19-true19),
            MAE_20=abs(pred20-true20),
            RMSE=np.sqrt(((pred19-true19)**2 + (pred20- true20)**2)/2),
            MedAPE=medape([true19, true20], [pred19, pred20]),
            MAPE_avg=mape([true19, true20], [pred19, pred20])
        )
        per_country.append(metrics)

        # -------- 绘图 --------
        yrs = list(range(2009, 2021))
        act = [pop[y] for y in yrs]
        pred = act.copy()
        pred[yrs.index(2019)] = pred19
        pred[yrs.index(2020)] = pred20

        plt.figure(figsize=(7.5,4.5))
        plt.plot(yrs, act, 'o-', label="Actual")
        plt.plot(yrs, pred, 'x--', label="Predicted")
        plt.title(f"Population — LocID {locid}")
        plt.xlabel("Year"); plt.ylabel("Population")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{locid}.png"))
        plt.close()

    # -------- 汇总到 CSV --------
    if not per_country:
        print("❌ 未生成任何结果，请检查数据与模型目录。")
        return

    pd.DataFrame(per_country).to_csv(result_csv, index=False)

    # -------- 聚合统计 + 直方图 --------
    mape_all = np.array([d["MAPE_avg"] for d in per_country])
    print("\n=== Aggregate statistics (based on MAPE_avg) ===")
    print(f" Countries evaluated : {len(mape_all)}")
    print(f" Mean   : {mape_all.mean():.2f}%")
    print(f" Std-dev: {mape_all.std(ddof=1):.2f}%")
    print(f" Min/Max: {mape_all.min():.2f}% / {mape_all.max():.2f}%")

    plt.figure(figsize=(6,4))
    plt.hist(mape_all, bins=12, edgecolor='k')
    plt.title("Distribution of MAPE_avg across countries")
    plt.xlabel("MAPE (%)"); plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "aggregate_mape_hist.png"))
    plt.close()

    # -------- 95 % CI (bootstrap) --------
    boot = np.random.choice(mape_all, size=(2000, len(mape_all)), replace=True)
    ci_low, ci_high = np.percentile(boot.mean(axis=1), [2.5, 97.5])
    print(f" 95 % CI for mean MAPE : [{ci_low:.2f} %, {ci_high:.2f} %]")
    print(f" Results written       : {result_csv}")
    print(f" Plots directory       : {plot_dir}")

# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",   default="train.csv")
    p.add_argument("--model_dir",   default="models")
    p.add_argument("--plot_dir",    default="plots")
    p.add_argument("--result_csv",  default="mape_results.csv")
    p.add_argument("--hidden_size", default=10, type=int)
    p.add_argument("--device",      default="cpu", choices=["cpu","cuda"])
    args = p.parse_args()

    evaluate(train_csv=args.train_csv,
             model_dir=args.model_dir,
             plot_dir=args.plot_dir,
             result_csv=args.result_csv,
             hidden_size=args.hidden_size,
             device=args.device)
