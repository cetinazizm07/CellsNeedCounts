"""Ablation table for the own CNN (validation set only; the test set is not used).

Reference: own_e15 (seed 655) plus the same configuration with seeds 656 and
657, which gives the seed-to-seed spread to judge whether an ablation's
change is larger than noise. Each ablation changes exactly one factor.

Usage: python -m src.ablation -> results/ablation.csv, results/ablation.md
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"

REFERENCE = ["own_e15", "own_e15_s656", "own_e15_s657"]
ABLATIONS = [
    ("own_abl_pool_avg", "Pooling", "sum -> global average + linear"),
    ("own_abl_no_c5", "Receptive field", "drop c5: RF 46 -> 30 px"),
    ("own_abl_48x64", "Input resolution", "96x128 -> 48x64"),
    ("own_abl_width_half", "Width", "16-32-64 -> 8-16-32 channels"),
    ("own_abl_amp", "Precision", "fp32 -> fp16 autocast (training)"),
]


def load(run):
    s = json.loads((RUNS / run / "summary.json").read_text())
    return {"run": run, "params": s["params"], "mmacs": s["macs"] / 1e6,
            "val_mae": s["best_val_mae"], "best_epoch": s["best_epoch"],
            "train_s": s["train_s"], "train_wh": s["train_kwh"] * 1000}


def main():
    ref = pd.DataFrame([load(r) for r in REFERENCE])
    mu, sd = ref.val_mae.mean(), ref.val_mae.std(ddof=1)
    wh_mu = ref.train_wh.mean()
    rows = [{"factor": "Reference (3 seeds)", "change": "own_e15 config", "run": ", ".join(REFERENCE),
             "params": int(ref.params.iloc[0]), "mmacs": ref.mmacs.iloc[0],
             "val_mae": mu, "val_mae_sd": sd, "train_wh": wh_mu,
             "train_s": ref.train_s.mean(), "d_mae": 0.0, "d_mae_in_sd": 0.0, "d_wh_pct": 0.0}]
    for run, factor, change in ABLATIONS:
        a = load(run)
        rows.append({"factor": factor, "change": change, "run": run, "params": a["params"],
                     "mmacs": a["mmacs"], "val_mae": a["val_mae"], "val_mae_sd": np.nan,
                     "train_wh": a["train_wh"], "train_s": a["train_s"],
                     "d_mae": a["val_mae"] - mu, "d_mae_in_sd": (a["val_mae"] - mu) / sd,
                     "d_wh_pct": 100 * (a["train_wh"] / wh_mu - 1)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "ablation.csv", index=False)

    lines = ["| Factor | Change | Params | MMACs | Val MAE | ΔMAE (in seed SDs) | Train Wh | ΔWh |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in df.itertuples():
        mae = f"{r.val_mae:.3f} ± {r.val_mae_sd:.3f}" if r.factor.startswith("Reference") else f"{r.val_mae:.3f}"
        dm = "–" if r.factor.startswith("Reference") else f"{r.d_mae:+.3f} ({r.d_mae_in_sd:+.1f})"
        dw = "–" if r.factor.startswith("Reference") else f"{r.d_wh_pct:+.0f} %"
        lines.append(f"| {r.factor} | {r.change} | {r.params:,} | {r.mmacs:.1f} | {mae} | {dm} | "
                     f"{r.train_wh:.3f} | {dw} |")
    md = "\n".join(lines) + "\n"
    (ROOT / "results" / "ablation.md").write_text(md)
    print(md)
    print("reference seeds:", ref[["run", "val_mae", "train_wh"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
