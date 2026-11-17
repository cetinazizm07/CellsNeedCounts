"""Test-set evaluation of a final model (run once per final model).

For a trained run (checkpoints/<run>.pt) this measures, with the same code
for every model:

1. Test metrics (MAE, RMSE, R2, bias) overall and per blur level, and the
   per-image predictions -> results/test/<run>.json, <run>_predictions.csv.
2. Inference latency per image at batch size 1 (median over 300 images after
   50 warm-up images; MPS synchronised per image).
3. Inference energy: batched (256) predictions over the test images repeated
   for >= 20 s inside its own measured run; energy per 1,000 predictions =
   measured kWh / N * 1000. It is gross energy (it includes the SoC's idle
   power during that time).
4. Model file size: bytes of the fp32 state dict saved with torch.save.

Guard: refuses to run again on a run that already has test results, so the
test set is used once per final model (override with --force and say why in
the report).

Usage: python -m src.evaluate --run own_e15
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analyze import count_params, model_macs
from src.energy import measure_run
from src.metrics import grouped_metrics, regression_metrics
from src.models import build_model
from src.train import CKPT, load_split, predict

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "test"


def load_trained(run: str, dev):
    ck = torch.load(CKPT / f"{run}.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    in_hw = tuple(int(v) for v in a["size"].split("x"))
    # Weights come from the checkpoint, so no ImageNet download is needed here.
    model = build_model(a["model"], in_hw, pretrained=False, **ck.get("kw", {}))
    model.load_state_dict(ck["model"])
    return model.to(dev).eval(), a, in_hw


def file_size_mb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


@torch.no_grad()
def latency_ms(model, x_u8, warmup=50, n=300):
    times = []
    for i in range(warmup + n):
        x = x_u8[i % x_u8.shape[0]:i % x_u8.shape[0] + 1].float() / 255.0
        torch.mps.synchronize()
        t = time.perf_counter()
        model(x)
        torch.mps.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


@torch.no_grad()
def throughput_run(model, x_u8, min_s=20.0, bs=256):
    n, t0 = 0, time.time()
    while time.time() - t0 < min_s:
        for i in range(0, x_u8.shape[0], bs):
            model(x_u8[i:i + bs].float() / 255.0)
            n += min(bs, x_u8.shape[0] - i)
        torch.mps.synchronize()
    return n, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / f"{args.run}.json"
    if out_json.exists() and not args.force:
        raise SystemExit(f"{out_json} exists: test set already used for this run.")

    dev = torch.device("mps")
    with measure_run(f"eval_{args.run}", phase="eval", model=args.run) as m_eval:
        model, a, in_hw = load_trained(args.run, dev)
        xte, yte, bte, meta = load_split("test", a["size"], dev)
        pred = predict(model, xte).cpu().numpy()
        y = yte.cpu().numpy()
        overall = regression_metrics(y, pred)
        by_blur = grouped_metrics(y, pred, bte)
        by_count = grouped_metrics(y, pred, y.astype(int))
        lat = latency_ms(model, xte)

    with measure_run(f"infer_{args.run}", phase="inference", model=args.run) as m_inf:
        n_pred, inf_s = throughput_run(model, xte)
    kwh_per_1000 = m_inf.row["meter_total_kwh"] / n_pred * 1000

    res = {
        "run": args.run, "model": a["model"], "size": a["size"],
        "params": count_params(model), "macs": model_macs(model.cpu(), in_hw),
        "file_mb": file_size_mb(model), "test": overall,
        "test_by_blur": by_blur, "test_by_count": by_count,
        "latency_ms_bs1": lat,
        "throughput_img_s_bs256": n_pred / inf_s,
        "inference_kwh_per_1000": kwh_per_1000,
        "inference_kg_co2eq_per_1000": m_inf.row["meter_kg_co2eq"] / n_pred * 1000,
        "eval_kwh": m_eval.row["meter_total_kwh"],
        "inference_bench_kwh": m_inf.row["meter_total_kwh"],
    }
    out_json.write_text(json.dumps(res, indent=2))
    pd.DataFrame({"filename": meta["filename"].to_numpy(), "count": y, "blur": bte,
                  "pred": pred}).to_csv(OUT / f"{args.run}_predictions.csv", index=False)
    t = overall
    print(f"{args.run}: test MAE {t['mae']:.3f}  RMSE {t['rmse']:.3f}  R2 {t['r2']:.4f}  "
          f"bias {t['bias']:+.3f} | {lat:.2f} ms/img (bs1) | "
          f"{kwh_per_1000 * 1000:.4f} Wh/1000 pred | {res['file_mb']:.2f} MB")


if __name__ == "__main__":
    main()
