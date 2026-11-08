"""Shared training script for every model (own CNN and standard backbones).

Same data, split, augmentation, loss, optimiser family, schedule and
validation code for all models; only the model and its hyperparameters differ.
The test split is never loaded here (see src/evaluate.py).

Design:
- Data: cached uint8 arrays are moved to the GPU once; batches are sliced and
  augmented on the GPU (no DataLoader workers, no repeated decoding).
- Augmentation: random horizontal and vertical flips. They keep the count
  exactly; 90-degree rotations would change the aspect ratio.
- Loss: Huber (smooth L1) on the count with delta = 1 cell: quadratic for
  sub-cell errors, linear (MAE-like) above, matching the MAE metric.
- Optimiser: AdamW, cosine decay with linear warm-up (1 epoch), per step.
- Stopping: keep the checkpoint with the best validation MAE; stop when it has
  not improved for `--patience` epochs, or at `--epochs`.

Usage examples:
  python -m src.train --model own --run-name own_v1
  python -m src.train --model own --subset 0.25 --epochs 5 --lr 3e-3 --run-name lr_own_3e-3
  python -m src.train --model resnet18 --pretrained --lr 1e-3 --run-name resnet18_ft
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.analyze import count_params, model_macs
from src.energy import measure_run
from src.metrics import regression_metrics
from src.models import build_model

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
RUNS = ROOT / "results" / "runs"
CKPT = ROOT / "checkpoints"


def load_split(split: str, size: str, device, subset: float = 1.0, seed: int = 655):
    """Return (images uint8 N x 1 x H x W on device, counts float32, blur, meta)."""
    meta = pd.read_csv(CACHE / "meta_w2.csv")
    h, w = (int(v) for v in size.split("x"))
    src_size = "96x128" if (h, w) in [(96, 128), (48, 64)] else "192x256"
    arr = np.load(CACHE / f"images_w2_{src_size}.npy", mmap_mode="r")
    idx = meta.index[meta.split == split].to_numpy()
    if subset < 1.0:
        # Subset by layout, so all blur levels of a chosen layout stay together.
        rng = np.random.default_rng(seed)
        layouts = meta.loc[idx, "layout_id"].unique()
        keep = set(rng.choice(layouts, size=max(1, int(len(layouts) * subset)), replace=False))
        idx = idx[meta.loc[idx, "layout_id"].isin(keep).to_numpy()]
    x = torch.from_numpy(np.ascontiguousarray(arr[idx])).unsqueeze(1)
    if (h, w) == (48, 64):  # 2x2 box average of the 96x128 cache
        x = F.avg_pool2d(x.float(), 2).round().to(torch.uint8)
    y = torch.tensor(meta.loc[idx, "count"].to_numpy(), dtype=torch.float32)
    return x.to(device), y.to(device), meta.loc[idx, "blur"].to_numpy(), meta.loc[idx]


def augment(x):
    n = x.shape[0]
    hf = torch.rand(n, 1, 1, 1, device=x.device) < 0.5
    vf = torch.rand(n, 1, 1, 1, device=x.device) < 0.5
    x = torch.where(hf, x.flip(-1), x)
    return torch.where(vf, x.flip(-2), x)


@torch.no_grad()
def predict(model, x_u8, bs=256):
    model.eval()
    out = []
    for i in range(0, x_u8.shape[0], bs):
        out.append(model(x_u8[i:i + bs].float() / 255.0))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="own")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--size", default="96x128")
    ap.add_argument("--widths", default="16,32,64", help="own CNN only")
    ap.add_argument("--pool", default="sum", help="own CNN only: sum | avg")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--subset", type=float, default=1.0, help="fraction of train layouts")
    ap.add_argument("--seed", type=int, default=655)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--phase", default="train", help="train | search | ablation")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    in_hw = tuple(int(v) for v in args.size.split("x"))
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(exist_ok=True)

    with measure_run(args.run_name, phase=args.phase, model=args.model,
                     notes=args.notes or json.dumps({k: v for k, v in vars(args).items()
                                                     if k not in ("run_name", "notes")})) as meas:
        t0 = time.time()
        xtr, ytr, _, _ = load_split("train", args.size, dev, args.subset, args.seed)
        xva, yva, bva, _ = load_split("val", args.size, dev)

        kw = {}
        if args.model == "own":
            xf = xtr.float() / 255.0
            kw = dict(widths=tuple(int(v) for v in args.widths.split(",")), pool=args.pool,
                      pixel_mean=xf.mean().item(), pixel_std=xf.std().item())
            del xf
        model = build_model(args.model, in_hw, pretrained=args.pretrained, **kw)
        n_params = count_params(model)
        macs = model_macs(model, in_hw)
        model = model.to(dev)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        steps_per_epoch = math.ceil(xtr.shape[0] / args.bs)
        total, warm = args.epochs * steps_per_epoch, steps_per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
                                                  else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

        best, best_epoch, history = float("inf"), -1, []
        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(xtr.shape[0], device=dev)
            loss_sum, n = 0.0, 0
            te = time.time()
            for i in range(0, xtr.shape[0], args.bs):
                b = perm[i:i + args.bs]
                x = augment(xtr[b]).float() / 255.0
                loss = F.huber_loss(model(x), ytr[b], delta=args.huber_delta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                loss_sum += loss.item() * b.numel()
                n += b.numel()
            pva = predict(model, xva)
            val = regression_metrics(yva.cpu().numpy(), pva.cpu().numpy())
            row = {"epoch": epoch, "train_loss": loss_sum / n, "val_mae": val["mae"],
                   "val_rmse": val["rmse"], "val_r2": val["r2"],
                   "lr": sched.get_last_lr()[0], "epoch_s": time.time() - te}
            history.append(row)
            improved = val["mae"] < best
            if improved:
                best, best_epoch = val["mae"], epoch
                torch.save({"model": model.state_dict(), "args": vars(args), "kw": kw},
                           CKPT / f"{args.run_name}.pt")
            print(f"ep {epoch:2d}  loss {row['train_loss']:.3f}  val MAE {val['mae']:.3f}  "
                  f"RMSE {val['rmse']:.3f}  {row['epoch_s']:.1f}s{'  *' if improved else ''}",
                  flush=True)
            if epoch - best_epoch >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epochs")
                break
        train_s = time.time() - t0

    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    summary = {"run_name": args.run_name, "args": vars(args), "params": n_params,
               "macs": macs, "best_val_mae": best, "best_epoch": best_epoch,
               "epochs_run": len(history), "train_s": train_s,
               "train_kwh": meas.row["meter_total_kwh"],
               "train_kg_co2eq": meas.row["meter_kg_co2eq"],
               "cc_kwh": meas.row["cc_energy_kwh"], "device": str(dev)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("params", "macs", "best_val_mae", "best_epoch",
                                              "epochs_run", "train_s", "train_kwh")}, indent=1))


if __name__ == "__main__":
    main()
