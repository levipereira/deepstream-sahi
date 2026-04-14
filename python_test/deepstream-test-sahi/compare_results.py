#!/usr/bin/env python3

################################################################################
# SPDX-License-Identifier: LicenseRef-NvidiaDeepStreamEULA
# Part of deepstream-sahi — subject to NVIDIA DeepStream SDK License Agreement:
# https://developer.nvidia.com/deepstream-eula
################################################################################

import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CLASS_COLUMNS = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor", "others"
]


def load_data(file_a: Path, file_b: Path):
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)
    return df_a, df_b


def plot_total_objects_over_frames(df_a, df_b, label_a, label_b):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_a["frame"], df_a["total_objects"], label=label_a, alpha=0.8, linewidth=1.2)
    ax.plot(df_b["frame"], df_b["total_objects"], label=label_b, alpha=0.8, linewidth=1.2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Total Objects")
    ax.set_title("Total Objects Detected per Frame")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_class_comparison_bar(df_a, df_b, label_a, label_b):
    means_a = df_a[CLASS_COLUMNS].mean()
    means_b = df_b[CLASS_COLUMNS].mean()
    x = np.arange(len(CLASS_COLUMNS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, means_a, width, label=label_a, alpha=0.9)
    ax.bar(x + width/2, means_b, width, label=label_b, alpha=0.9)
    ax.set_xlabel("Class")
    ax.set_ylabel("Mean Detections per Frame")
    ax.set_title("Mean Detections per Class")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_COLUMNS, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_total_objects_histogram(df_a, df_b, label_a, label_b):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_a["total_objects"], bins=30, alpha=0.6, label=label_a,
            color="C0", edgecolor="black", linewidth=0.5)
    ax.hist(df_b["total_objects"], bins=30, alpha=0.6, label=label_b,
            color="C1", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Total Objects")
    ax.set_ylabel("Frequency (frames)")
    ax.set_title("Total Objects Distribution per Frame")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_difference_over_frames(df_a, df_b, label_a, label_b):
    diff = df_a["total_objects"].values - df_b["total_objects"].values
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = np.where(diff >= 0, "C0", "C1")
    ax.bar(df_a["frame"], diff, color=colors, alpha=0.7, width=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel(f"Difference ({label_a} - {label_b})")
    ax.set_title("Total Objects Difference per Frame")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_top_classes_over_time(df_a, df_b, label_a, label_b, top_n=5):
    totals = df_a[CLASS_COLUMNS].sum()
    top_classes = totals.nlargest(top_n).index.tolist()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, df, label in [(axes[0], df_a, label_a), (axes[1], df_b, label_b)]:
        for col in top_classes:
            ax.plot(df["frame"], df[col], label=col, alpha=0.8)
        ax.set_ylabel("Detections")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Frame")
    fig.suptitle(f"Top {top_n} Classes Over Time", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def build_stats(df, label):
    total = df["total_objects"]
    per_class = df[CLASS_COLUMNS].mean()
    return {
        "label": label,
        "frames": len(df),
        "mean": total.mean(),
        "std": total.std(),
        "min": total.min(),
        "max": total.max(),
        "median": total.median(),
        "per_class": per_class,
    }


def generate_report(stats_a, stats_b, plot_files, output_dir):
    sa, sb = stats_a, stats_b
    diff_mean = sa["mean"] - sb["mean"]
    diff_pct = (diff_mean / sb["mean"] * 100) if sb["mean"] != 0 else 0

    lines = [
        f"# Detection Comparison Report",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Overview",
        f"",
        f"| | **{sa['label']}** | **{sb['label']}** |",
        f"|---|---|---|",
        f"| Frames analyzed | {sa['frames']} | {sb['frames']} |",
        f"| Mean objects/frame | {sa['mean']:.1f} | {sb['mean']:.1f} |",
        f"| Std deviation | {sa['std']:.1f} | {sb['std']:.1f} |",
        f"| Median objects/frame | {sa['median']:.0f} | {sb['median']:.0f} |",
        f"| Min objects/frame | {sa['min']} | {sb['min']} |",
        f"| Max objects/frame | {sa['max']} | {sb['max']} |",
        f"",
        f"**Mean difference ({sa['label']} - {sb['label']}):** "
        f"{diff_mean:+.1f} objects/frame ({diff_pct:+.1f}%)",
        f"",
        f"## Per-Class Mean Detections",
        f"",
        f"| Class | **{sa['label']}** | **{sb['label']}** | Diff |",
        f"|---|---|---|---|",
    ]

    for cls in CLASS_COLUMNS:
        va = sa["per_class"][cls]
        vb = sb["per_class"][cls]
        d = va - vb
        lines.append(f"| {cls} | {va:.2f} | {vb:.2f} | {d:+.2f} |")

    lines += [
        f"",
        f"## Charts",
        f"",
    ]

    chart_titles = {
        "01_total_objects_over_frames.png": "Total Objects Detected per Frame",
        "02_class_comparison_bar.png": "Mean Detections per Class",
        "03_total_objects_histogram.png": "Total Objects Distribution",
        "04_difference_over_frames.png": "Detection Difference per Frame",
        "05_top_classes_over_time.png": "Top Classes Over Time",
    }

    for fname in plot_files:
        title = chart_titles.get(fname, fname)
        lines.append(f"### {title}")
        lines.append(f"")
        lines.append(f"![{title}]({fname})")
        lines.append(f"")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def print_summary(df_a, df_b, label_a, label_b):
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{label_a}:")
    print(f"  - Mean total_objects: {df_a['total_objects'].mean():.1f}")
    print(f"  - Std dev: {df_a['total_objects'].std():.1f}")
    print(f"  - Min/Max: {df_a['total_objects'].min()}/{df_a['total_objects'].max()}")
    print(f"\n{label_b}:")
    print(f"  - Mean total_objects: {df_b['total_objects'].mean():.1f}")
    print(f"  - Std dev: {df_b['total_objects'].std():.1f}")
    print(f"  - Min/Max: {df_b['total_objects'].min()}/{df_b['total_objects'].max()}")
    diff_mean = df_a['total_objects'].mean() - df_b['total_objects'].mean()
    print(f"\nMean difference ({label_a} - {label_b}): {diff_mean:.1f} objects per frame")
    print("="*60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two SAHI detection result CSVs")
    parser.add_argument("file_a", type=Path, help="First CSV file")
    parser.add_argument("file_b", type=Path, help="Second CSV file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory (default: comparison_<a>_vs_<b>)")
    parser.add_argument("-a", "--label-a", type=str, default=None,
                        help="Label for file_a (default: filename stem)")
    parser.add_argument("-b", "--label-b", type=str, default=None,
                        help="Label for file_b (default: filename stem)")
    return parser.parse_args()


def main():
    args = parse_args()
    label_a = args.label_a or args.file_a.stem
    label_b = args.label_b or args.file_b.stem
    base = Path(__file__).parent / "results"
    output_dir = args.output or base / f"comparison_{label_a}_vs_{label_b}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df_a, df_b = load_data(args.file_a, args.file_b)
    min_frames = min(len(df_a), len(df_b))
    df_a = df_a.head(min_frames)
    df_b = df_b.head(min_frames)

    print_summary(df_a, df_b, label_a, label_b)

    plot_files = []
    plots = [
        ("01_total_objects_over_frames.png", lambda: plot_total_objects_over_frames(df_a, df_b, label_a, label_b)),
        ("02_class_comparison_bar.png", lambda: plot_class_comparison_bar(df_a, df_b, label_a, label_b)),
        ("03_total_objects_histogram.png", lambda: plot_total_objects_histogram(df_a, df_b, label_a, label_b)),
        ("04_difference_over_frames.png", lambda: plot_difference_over_frames(df_a, df_b, label_a, label_b)),
        ("05_top_classes_over_time.png", lambda: plot_top_classes_over_time(df_a, df_b, label_a, label_b, 5)),
    ]
    for filename, plot_func in plots:
        print(f"Generating {filename}...")
        fig = plot_func()
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plot_files.append(filename)

    stats_a = build_stats(df_a, label_a)
    stats_b = build_stats(df_b, label_b)
    report_path = generate_report(stats_a, stats_b, plot_files, output_dir)

    print(f"\nPlots saved to: {output_dir}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
