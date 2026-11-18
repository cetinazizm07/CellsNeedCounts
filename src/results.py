"""Build results/results.csv (handout §3.5 table) and the error vs CO2 plot.

Per-model footprint attribution (logs/runs.csv, by run_name):
  train  = the final training run of that model
  project = every run that belongs to the model family: its LR scans, earlier
            discarded configurations, the final run, its test evaluation and
            its inference benchmark. Shared runs (data prep, mean baseline)
            are reported on their own row and in the project total.

Usage: python -m src.results
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = pd.read_csv(ROOT / "logs" / "runs.csv")
OUT = ROOT / "results"

# label, final run, regex for every run of the family (None: a tested extra
# configuration whose runs are already counted in another row's family)
FINAL_OWN = "own_green_48x64"
MODELS = [
    ("Own CNN final (8-16-32, no c5, fp16, 48x64)", FINAL_OWN, r"^(?:search_)?own_|^(?:eval|infer|ext)_own_"),
    ("Own CNN reference (16-32-64, 96x128)", "own_e15", None),
    ("MobileNetV3-S (ImageNet FT)", "mnv3s_ft_e15", r"^(?:search_)?mnv3s_ft|^(?:eval|infer|ext)_mnv3s_ft"),
    ("ResNet-18 (ImageNet FT)", "resnet18_ft_e15", r"^(?:search_)?resnet18_ft|^(?:eval|infer|ext)_resnet18_ft"),
    ("MobileNetV3-S (scratch)", "mnv3s_scratch_e15", r"^(?:search_)?mnv3s_scratch|^(?:eval|infer|ext)_mnv3s_scratch"),
    ("ResNet-18 (ImageNet, frozen backbone)", "resnet18_probe_e15", r"^resnet18_probe|^(?:eval|infer|ext)_resnet18_probe"),
]
SHARED = r"^(?:data_prep|baseline_|ext_prep)"


def family(regex):
    return RUNS[RUNS.run_name.str.contains(regex, regex=True)]


def main():
    rows = []
    base = json.loads((OUT / "baseline.json").read_text())["mean"]
    rows.append({"model": "Mean baseline", "run": "baseline", "params": 0, "mflops_per_img": 0.0,
                 "test_mae": base["test"]["mae"], "test_rmse": base["test"]["rmse"],
                 "test_r2": base["test"]["r2"], "train_kwh": None, "train_kg_co2eq": None,
                 "project_kwh": None, "project_kg_co2eq": None, "n_runs": None,
                 "ms_per_img": None, "file_mb": 0.0, "wh_per_1000_pred": None})
    claimed = set()
    for label, run, regex in MODELS:
        t = json.loads((OUT / "test" / f"{run}.json").read_text())
        fam = family(regex) if regex else None
        if fam is not None:
            claimed |= set(fam.run_name)
        final = RUNS[RUNS.run_name == run].iloc[-1]
        rows.append({
            "model": label, "run": run, "params": t["params"],
            "mflops_per_img": 2 * t["macs"] / 1e6,
            "test_mae": t["test"]["mae"], "test_rmse": t["test"]["rmse"], "test_r2": t["test"]["r2"],
            "train_kwh": final.meter_total_kwh, "train_kg_co2eq": final.meter_kg_co2eq,
            "project_kwh": fam.meter_total_kwh.sum() if fam is not None else None,
            "project_kg_co2eq": fam.meter_kg_co2eq.sum() if fam is not None else None,
            "n_runs": len(fam) if fam is not None else None, "ms_per_img": t["latency_ms_bs1"], "file_mb": t["file_mb"],
            "wh_per_1000_pred": t["inference_kwh_per_1000"] * 1000,
        })
    shared = family(SHARED)
    fams = [set(family(rx).run_name) for _, _, rx in MODELS if rx] + [set(shared.run_name)]
    assert sum(len(f) for f in fams) == len(set().union(*fams)), "a run is in two families"
    claimed |= set(shared.run_name)
    unclaimed = sorted(set(RUNS.run_name) - claimed)
    assert not unclaimed, f"runs not attributed to any model: {unclaimed}"

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)

    best_std = df[df.model.str.contains("ResNet|MobileNet")].test_mae.min()
    thr = min(1.15 * best_std, 0.6 * base["test"]["mae"])
    # The criterion is defined for the own model only (handout §3.5).
    df["acceptable"] = None
    own = df.run.str.startswith("own_")
    df.loc[own, "acceptable"] = df.loc[own, "test_mae"] <= thr
    df.to_csv(OUT / "results.csv", index=False)
    summary = {"mae_best_standard": best_std, "threshold_1.15x_best": 1.15 * best_std,
               "threshold_0.6x_mean": 0.6 * base["test"]["mae"],
               "final_own_run": FINAL_OWN,
               "own_acceptable": bool(df.loc[df.run == FINAL_OWN, "acceptable"].iloc[0]),
               "own_margin_to_threshold": float(thr - df.loc[df.run == FINAL_OWN, "test_mae"].iloc[0]),
               "own_reference_acceptable": bool(df.loc[df.run == "own_e15", "acceptable"].iloc[0]),
               "shared_kwh": shared.meter_total_kwh.sum(),
               "project_total_kwh": RUNS.meter_total_kwh.sum(),
               "project_total_kg_co2eq": RUNS.meter_kg_co2eq.sum(),
               "n_runs_total": len(RUNS)}
    (OUT / "acceptability.json").write_text(json.dumps(summary, indent=2))

    show = df.copy()
    for c in ("train_kwh", "project_kwh"):
        show[c.replace("kwh", "wh")] = show.pop(c) * 1000
    for c in ("train_kg_co2eq", "project_kg_co2eq"):
        show[c.replace("kg", "g")] = show.pop(c) * 1000
    pd.set_option("display.width", 250)
    print(show[["model", "params", "mflops_per_img", "test_mae", "test_rmse", "test_r2",
                "train_wh", "train_g_co2eq", "project_wh", "project_g_co2eq", "n_runs",
                "ms_per_img", "file_mb", "wh_per_1000_pred", "acceptable"]].round(4).to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
