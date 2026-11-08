"""Trivial baselines: predict the mean / median of the training targets.

Writes results/baseline.json. The test set is touched once here, as for every
final model.

Usage: python -m src.baseline
"""
import json
from pathlib import Path

import pandas as pd

from src.energy import measure_run
from src.metrics import grouped_metrics, regression_metrics

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "cache" / "meta_w2.csv"
OUT = ROOT / "results" / "baseline.json"


def main():
    df = pd.read_csv(META)
    train = df[df.split == "train"]
    out = {}
    with measure_run("baseline_mean_median", phase="eval", model="baseline"):
        for name, value in [("mean", train["count"].mean()),
                            ("median", train["count"].median())]:
            out[name] = {"prediction": float(value)}
            for split in ("val", "test"):
                part = df[df.split == split]
                pred = [value] * len(part)
                out[name][split] = regression_metrics(part["count"], pred)
                out[name][f"{split}_by_blur"] = grouped_metrics(part["count"], pred, part["blur"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    for name in out:
        t = out[name]["test"]
        print(f"{name:6s} pred={out[name]['prediction']:.2f}  test MAE {t['mae']:.3f}  "
              f"RMSE {t['rmse']:.3f}  R2 {t['r2']:.4f}")


if __name__ == "__main__":
    main()
