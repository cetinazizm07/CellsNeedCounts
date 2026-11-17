"""Frozen-backbone transfer learning (linear probe) for a torchvision model.

The ImageNet backbone is never updated (eval mode, ImageNet BN statistics);
only the 1-output linear head is trained. Because the backbone is fixed, its
features are computed once:

- train: for each image, the features of all 4 flip variants (none, h, v, hv).
  Each epoch picks one variant per image at random, which is exactly the flip
  augmentation used by src/train.py, at no extra backbone cost.
- val: features of the unflipped images.

Then the head is trained with the same loss (Huber, delta 1), optimiser
(AdamW), warm-up + cosine schedule, batch size and epoch budget as the other
models. The head LR is picked on validation from a small grid inside this
same measured run (the grid is cheap once features exist).

The saved checkpoint is a normal TorchvisionRegressor state dict (backbone +
trained head), so evaluation code is identical to the other models.

Usage: python -m src.train_probe --model resnet18 --run-name resnet18_probe_e15
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analyze import count_params, model_macs
from src.energy import measure_run
from src.metrics import regression_metrics
from src.models import TRAIN_COUNT_MEAN, build_model
from src.train import CKPT, RUNS, load_split

HEAD_ATTR = {"resnet18": ("fc",), "mobilenet_v3_small": ("classifier", -1)}


def get_head(net, arch):
    path = HEAD_ATTR[arch]
    return getattr(net, path[0]) if len(path) == 1 else getattr(net, path[0])[path[1]]


def set_head(net, arch, module):
    path = HEAD_ATTR[arch]
    if len(path) == 1:
        setattr(net, path[0], module)
    else:
        getattr(net, path[0])[path[1]] = module


@torch.no_grad()
def features(model, arch, x_u8, flips=((),), bs=256):
    """Backbone output (input of the head) for each flip variant: V x N x D."""
    head = get_head(model.net, arch)
    set_head(model.net, arch, nn.Identity())
    model.eval()
    out = []
    for dims in flips:
        f = []
        for i in range(0, x_u8.shape[0], bs):
            x = x_u8[i:i + bs].float() / 255.0
            if dims:
                x = x.flip(dims)
            # Same preprocessing as TorchvisionRegressor.forward, head removed.
            f.append(model.net((x.expand(-1, 3, -1, -1) - model.mean) / model.std))
        out.append(torch.cat(f))
    set_head(model.net, arch, head)
    return torch.stack(out)


def train_head(ftr, ytr, fva, yva, lr, epochs, bs, wd, delta, seed):
    torch.manual_seed(seed)
    dev = ftr.device
    head = nn.Linear(ftr.shape[-1], 1).to(dev)
    nn.init.normal_(head.weight, std=1e-3)
    nn.init.constant_(head.bias, TRAIN_COUNT_MEAN)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = ftr.shape[1]
    spe = math.ceil(n / bs)
    total, warm = epochs * spe, spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
                                              else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    best, best_state, best_epoch, hist = float("inf"), None, -1, []
    for epoch in range(1, epochs + 1):
        variant = torch.randint(0, ftr.shape[0], (n,), device=dev)  # random flip per image
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = F.huber_loss(head(ftr[variant[b], b]).squeeze(1), ytr[b], delta=delta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            mae = regression_metrics(yva.cpu().numpy(), head(fva).squeeze(1).cpu().numpy())["mae"]
        hist.append({"epoch": epoch, "val_mae": mae})
        if mae < best:
            best, best_epoch = mae, epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    return best, best_epoch, best_state, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18", choices=list(HEAD_ATTR))
    ap.add_argument("--size", default="96x128")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lrs", default="1e-3,3e-3,1e-2,3e-2")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=655)
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    in_hw = tuple(int(v) for v in args.size.split("x"))
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(exist_ok=True)

    with measure_run(args.run_name, phase="train", model=f"{args.model}_probe",
                     notes=json.dumps({k: v for k, v in vars(args).items() if k != "run_name"})) as meas:
        t0 = time.time()
        xtr, ytr, _, _ = load_split("train", args.size, dev)
        xva, yva, _, _ = load_split("val", args.size, dev)
        model = build_model(args.model, in_hw, pretrained=True)
        n_params = count_params(model)
        macs = model_macs(model, in_hw)
        model = model.to(dev)
        for p in model.parameters():
            p.requires_grad_(False)
        tf = time.time()
        ftr = features(model, args.model, xtr, flips=((), (-1,), (-2,), (-1, -2)))
        fva = features(model, args.model, xva)[0]
        feat_s = time.time() - tf

        results = {}
        for lr in (float(v) for v in args.lrs.split(",")):
            best, ep, state, hist = train_head(ftr, ytr, fva, yva, lr, args.epochs, args.bs,
                                               args.wd, args.huber_delta, args.seed)
            results[lr] = (best, ep, state, hist)
            print(f"head lr {lr:g}: best val MAE {best:.3f} @ep {ep}", flush=True)
        lr_best = min(results, key=lambda k: results[k][0])
        best, best_epoch, state, history = results[lr_best]
        head = get_head(model.net, args.model)
        head.load_state_dict(state)
        torch.save({"model": model.state_dict(), "args": vars(args), "kw": {}},
                   CKPT / f"{args.run_name}.pt")
        train_s = time.time() - t0

    with open(run_dir / "history.csv", "w") as f:
        f.write("epoch,train_loss,val_mae\n")
        for h in history:
            f.write(f"{h['epoch']},,{h['val_mae']}\n")
    summary = {"run_name": args.run_name, "args": vars(args), "params": n_params,
               "trainable_params": count_params(head), "macs": macs,
               "best_val_mae": best, "best_epoch": best_epoch, "head_lr": lr_best,
               "lr_grid": {str(k): v[0] for k, v in results.items()},
               "epochs_run": args.epochs, "feature_s": feat_s, "train_s": train_s,
               "train_kwh": meas.row["meter_total_kwh"],
               "train_kg_co2eq": meas.row["meter_kg_co2eq"],
               "cc_kwh": meas.row["cc_energy_kwh"], "device": str(dev)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("trainable_params", "best_val_mae", "head_lr",
                                              "feature_s", "train_s", "train_kwh")}, indent=1))


if __name__ == "__main__":
    main()
