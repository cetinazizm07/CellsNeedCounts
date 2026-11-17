"""Error vs CO2 plot (handout §3.5): one point per model, Pareto front marked.

Two panels with a shared y axis: CO2 of the final training run, and CO2 of the
whole project attributed to that model (results/results.csv). Log scale on
CO2. Points on the Pareto front (no other model has both lower MAE and lower
CO2) are blue and joined by the front line; dominated points are gray. Every
point is labelled directly, so identity never depends on colour. The dashed
line is the acceptability threshold for the own model.

Usage: python -m src.plot  -> results/error_vs_co2.{png,pdf}
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FRONT, DOMINATED = "#2a78d6", "#9a9893"
SHORT = {
    "Own CNN (scratch)": "Own CNN",
    "MobileNetV3-S (ImageNet FT)": "MobileNetV3-S FT",
    "ResNet-18 (ImageNet FT)": "ResNet-18 FT",
    "MobileNetV3-S (scratch)": "MobileNetV3-S scratch",
    "ResNet-18 (ImageNet, frozen backbone)": "ResNet-18 frozen",
}


def pareto(df, x, y):
    pts = df.sort_values([x, y])
    front, best = [], float("inf")
    for i, r in pts.iterrows():
        if r[y] < best:
            front.append(i)
            best = r[y]
    return front


# Label offsets in points (dx, dy, horizontal alignment), per panel, chosen so
# the four points near MAE 1.5 do not collide.
OFFSETS = {
    "train_kg_co2eq": {"Own CNN": (-9, -14, "right"), "MobileNetV3-S FT": (6, -15, "left"),
                       "MobileNetV3-S scratch": (9, 1, "left"), "ResNet-18 FT": (9, -2, "left"),
                       "ResNet-18 frozen": (9, 4, "left")},
    "project_kg_co2eq": {"Own CNN": (-9, -14, "right"), "MobileNetV3-S FT": (2, -16, "center"),
                         "MobileNetV3-S scratch": (9, 1, "left"), "ResNet-18 FT": (9, -2, "left"),
                         "ResNet-18 frozen": (9, 4, "left")},
}


def panel(ax, df, xcol, title, thr):
    front = pareto(df, xcol, "test_mae")
    f = df.loc[front].sort_values(xcol)
    ax.plot(f[xcol], f.test_mae, color=FRONT, lw=2, zorder=2, drawstyle="steps-post",
            label="Pareto front")
    for i, r in df.iterrows():
        on = i in front
        ax.scatter(r[xcol], r.test_mae, s=90, color=FRONT if on else DOMINATED,
                   edgecolor="white", linewidth=2, zorder=3)
        dx, dy, ha = OFFSETS[xcol][SHORT[r.model]]
        ax.annotate(SHORT[r.model], (r[xcol], r.test_mae), xytext=(dx, dy), ha=ha, va="center",
                    textcoords="offset points", fontsize=8.5, color=TEXT if on else TEXT2)
    ax.axhline(thr, color=TEXT2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"acceptable own-model MAE ≤ {thr:.3f}", (0, thr), xycoords=("axes fraction", "data"),
                xytext=(4, 5), textcoords="offset points", ha="left", fontsize=8, color=TEXT2)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    ax.set_xlabel("kg CO$_2$eq (log scale)", fontsize=9, color=TEXT2)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=TEXT2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    lo, hi = df[xcol].min(), df[xcol].max()
    ax.set_xlim(lo / 1.6, hi * 3.0)


def main():
    df = pd.read_csv(RES / "results.csv")
    df = df[df.run != "baseline"].reset_index(drop=True)
    acc = json.loads((RES / "acceptability.json").read_text())
    thr = min(acc["threshold_1.15x_best"], acc["threshold_0.6x_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0), sharey=True)
    panel(axes[0], df, "train_kg_co2eq", "Final training run", thr)
    panel(axes[1], df, "project_kg_co2eq", "Whole project (all runs of the model)", thr)
    axes[0].set_ylabel("Test MAE (cells)", fontsize=9, color=TEXT2)
    axes[0].set_ylim(1.25, 4.1)
    axes[1].legend(loc="upper right", fontsize=8, frameon=False, labelcolor=TEXT2)
    fig.suptitle(f"BBBC005 test error vs carbon footprint  ·  mean baseline MAE "
                 f"{acc['threshold_0.6x_mean'] / 0.6:.2f} (off scale)",
                 fontsize=10.5, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"error_vs_co2.{ext}", dpi=200, facecolor="white")
    print("wrote", RES / "error_vs_co2.png")


if __name__ == "__main__":
    main()
