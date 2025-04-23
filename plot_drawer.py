#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch-plot every scalar tag in TensorBoard logs, preserving directory tree.

Example:
    python plot_all_tensorboard.py --logdir runs --outdir tb_plots
"""

import os, argparse, glob
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator as EA


def load_run_scalars(run_dir: str):
    """
    返回 dict{prefix: [(subtag, steps, values), ...]}
      prefix  : 'MAE' / 'Loss' / ...
      subtag  : 'train' / 'test' / ...
    """
    acc = EA.EventAccumulator(run_dir, size_guidance={"scalars": 0})
    try:
        acc.Reload()
    except Exception as e:        # 某些损坏日志跳过
        print(f"⚠️  {run_dir}: {e}")
        return {}

    tag_map = {}
    for full_tag in acc.Tags().get("scalars", []):
        parts = full_tag.split("/")
        prefix = parts[0]                 # 前缀
        subtag = parts[1] if len(parts) > 1 else "value"
        events = acc.Scalars(full_tag)
        steps  = [e.step  for e in events]
        values = [e.value for e in events]
        tag_map.setdefault(prefix, []).append((subtag, steps, values))
    return tag_map


def plot_prefix(run_dir: str, out_dir: str, prefix: str, curves):
    """
    将同一 prefix 的多条曲线画在一张图并保存
    """
    plt.figure(figsize=(7, 4))
    for subtag, steps, vals in curves:
        plt.plot(steps, vals, label=subtag, alpha=.8)
    plt.title(f"{prefix} — {Path(run_dir).name}")
    plt.xlabel("Step / Epoch")
    plt.ylabel(prefix)
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{prefix}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"✓ Saved {fname}")


def main(logdir: str, outdir: str):
    # 遍历所有含 events 文件的目录
    run_dirs = []
    for root, _, files in os.walk(logdir):
        if any(fn.startswith("events") for fn in files):
            run_dirs.append(root)

    if not run_dirs:
        print("❌ 未在 logdir 中找到事件文件")
        return

    for run in sorted(run_dirs):
        rel_path = os.path.relpath(run, logdir)        # 例：1/xyz
        out_run  = os.path.join(outdir, rel_path)
        tag_map  = load_run_scalars(run)
        if not tag_map:
            continue
        for prefix, curves in tag_map.items():
            plot_prefix(run, out_run, prefix, curves)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir",  default="runs",     help="根日志目录")
    ap.add_argument("--outdir",  default="tb_plots", help="输出图片根目录")
    args = ap.parse_args()
    main(args.logdir, args.outdir)
