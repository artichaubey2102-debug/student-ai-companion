"""
generate_charts.py
------------------
Reads evaluation result CSVs and produces publication-quality charts
for inclusion in the thesis report.

Output images saved to evaluation/charts/
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_CHART_DIR = os.path.join(os.path.dirname(__file__), "charts")


def _style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
    })


def chart_chunking():
    path = os.path.join(_DATA_DIR, "chunking_summary.csv")
    if not os.path.exists(path):
        print("chunking_summary.csv not found. Run evaluate_chunking.py first.")
        return

    df = pd.read_csv(path)
    metrics = ["context_precision", "context_recall", "top1_score"]
    labels = ["Context Precision", "Context Recall", "Top-1 Score"]
    strategies = df["strategy"].tolist()
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#2563EB", "#16A34A", "#D97706"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (strat, color) in enumerate(zip(strategies, colors)):
        row = df[df["strategy"] == strat].iloc[0]
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=strat.capitalize(),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Chunking Strategy Comparison", fontweight="bold", pad=12)
    ax.legend(title="Strategy", framealpha=0.7)
    plt.tight_layout()
    out = os.path.join(_CHART_DIR, "chunking_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Avg chunk length bar
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.bar(df["strategy"], df["mean_chunk_length"],
            color=colors[:len(df)], alpha=0.85)
    for i, (strat, val) in enumerate(zip(df["strategy"], df["mean_chunk_length"])):
        ax2.text(i, val + 5, f"{int(val)}", ha="center", fontsize=9)
    ax2.set_ylabel("Average chunk length (chars)")
    ax2.set_title("Average Chunk Length by Strategy", fontweight="bold", pad=10)
    plt.tight_layout()
    out2 = os.path.join(_CHART_DIR, "chunk_length_comparison.png")
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")


def chart_retrieval():
    path = os.path.join(_DATA_DIR, "retrieval_summary.csv")
    if not os.path.exists(path):
        print("retrieval_summary.csv not found. Run evaluate_retrieval.py first.")
        return

    df = pd.read_csv(path)
    metrics = ["precision_at_k", "mrr", "ndcg", "diversity"]
    labels = ["Precision@5", "MRR", "NDCG", "Diversity"]
    methods = df["method"].tolist()
    x = np.arange(len(metrics))
    width = 0.35
    colors = ["#2563EB", "#D97706"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (method, color) in enumerate(zip(methods, colors)):
        row = df[df["method"] == method].iloc[0]
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width,
                      label=method.upper(), color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Method Comparison: Cosine vs MMR", fontweight="bold", pad=12)
    ax.legend(title="Method", framealpha=0.7)
    plt.tight_layout()
    out = os.path.join(_CHART_DIR, "retrieval_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    os.makedirs(_CHART_DIR, exist_ok=True)
    _style()
    print("Generating chunking charts...")
    chart_chunking()
    print("Generating retrieval charts...")
    chart_retrieval()
    print("Done. Charts saved to evaluation/charts/")


if __name__ == "__main__":
    main()
