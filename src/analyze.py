"""Per-layer analysis of the own CNN and size/compute of any model.

For the own CNN: one row per layer with output shape, parameters, receptive
field (RF), effective stride ("jump") and MACs. Totals are cross-checked with
thop.

RF recursion for a layer with kernel k and stride s (input RF r, jump j):
    r_out = r + (k - 1) * j,   j_out = j * s      (r = j = 1 at the input)
MACs of a conv: H_out * W_out * C_out * (C_in / groups) * k_h * k_w.
BatchNorm / ReLU / pooling / softplus / sum are elementwise and listed with
0 MACs (they are a small, separate cost; FLOPs ~= 2 * MACs for the convs).

Usage: python -m src.analyze [--size 96x128] [--widths 16,32,64] [--pool sum]
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.models import OwnCountCNN, build_model

ROOT = Path(__file__).resolve().parent.parent


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def conv_macs(m: nn.Conv2d, out):
    h, w = out.shape[-2:]
    kh, kw = m.kernel_size
    return h * w * m.out_channels * (m.in_channels // m.groups) * kh * kw


def model_macs(model, in_hw):
    """MACs of all Conv2d and Linear layers for one 1-channel image."""
    total = 0
    hooks = []

    def hook(m, inp, out):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            total += conv_macs(m, out)
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, *in_hw))
    for h in hooks:
        h.remove()
    return total


def own_layer_table(model: OwnCountCNN, in_hw):
    rows = []
    x = torch.zeros(1, 1, *in_hw)
    r, j = 1, 1
    rows.append({"layer": "input", "type": "-", "out_shape": tuple(x.shape[1:]),
                 "params": 0, "rf": r, "jump": j, "macs": 0})
    names = {}
    ci = pi = 0
    for block in model.features:
        if isinstance(block, nn.MaxPool2d):
            pi += 1
            names[block] = f"p{pi}"
        else:
            ci += 1
            names[block] = f"c{ci}"
    model.eval()
    with torch.no_grad():
        for block in model.features:
            if isinstance(block, nn.MaxPool2d):
                k, s = block.kernel_size, block.stride
                x = block(x)
                r, j = r + (k - 1) * j, j * s
                rows.append({"layer": names[block], "type": f"MaxPool {k}x{k} /{s}",
                             "out_shape": tuple(x.shape[1:]), "params": 0,
                             "rf": r, "jump": j, "macs": 0})
                continue
            conv, bn, _ = block
            y = conv(x)
            k, s = conv.kernel_size[0], conv.stride[0]
            r, j = r + (k - 1) * j, j * s
            rows.append({"layer": names[block] + ".conv",
                         "type": f"Conv {k}x{k} {conv.in_channels}->{conv.out_channels}, pad {conv.padding[0]}",
                         "out_shape": tuple(y.shape[1:]), "params": count_params(conv),
                         "rf": r, "jump": j, "macs": conv_macs(conv, y)})
            x = block(x)
            rows.append({"layer": names[block] + ".bn+relu", "type": "BatchNorm + ReLU",
                         "out_shape": tuple(x.shape[1:]), "params": count_params(bn),
                         "rf": r, "jump": j, "macs": 0})
        if model.pool == "sum":
            y = model.head(x)
            rows.append({"layer": "head", "type": "Conv 1x1 -> 1 + softplus (density)",
                         "out_shape": tuple(y.shape[1:]), "params": count_params(model.head),
                         "rf": r, "jump": j, "macs": conv_macs(model.head, y)})
            rows.append({"layer": "sum", "type": "Sum over density map",
                         "out_shape": (1,), "params": 0, "rf": "whole image", "jump": "-",
                         "macs": 0})
        else:
            rows.append({"layer": "gap", "type": "Global average pool",
                         "out_shape": (x.shape[1],), "params": 0, "rf": "whole image",
                         "jump": "-", "macs": 0})
            rows.append({"layer": "head", "type": f"Linear {x.shape[1]}->1",
                         "out_shape": (1,), "params": count_params(model.head),
                         "rf": "whole image", "jump": "-",
                         "macs": model.head.in_features})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="96x128")
    ap.add_argument("--widths", default="16,32,64")
    ap.add_argument("--pool", default="sum")
    args = ap.parse_args()
    in_hw = tuple(int(v) for v in args.size.split("x"))
    widths = tuple(int(v) for v in args.widths.split(","))

    model = OwnCountCNN(in_hw=in_hw, widths=widths, pool=args.pool)
    rows = own_layer_table(model, in_hw)
    total_p = count_params(model)
    total_m = sum(r["macs"] for r in rows)
    assert total_p == sum(r["params"] for r in rows), "params do not add up"

    from thop import profile
    thop_macs, thop_params = profile(model, inputs=(torch.zeros(1, 1, *in_hw),), verbose=False)

    tag = f"own_{args.pool}_w{'-'.join(map(str, widths))}_{args.size}"
    out = ROOT / "results" / "layers"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"| layer | type | output (C,H,W) | params | RF (px) | jump | MACs |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['layer']} | {r['type']} | {r['out_shape']} | {r['params']:,} | "
                     f"{r['rf']} | {r['jump']} | {r['macs']:,} |")
    lines.append(f"| **total** | | | **{total_p:,}** | | | **{total_m:,}** |")
    (out / f"{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nparams {total_p:,}  MACs {total_m / 1e6:.2f} M  (~{2 * total_m / 1e6:.1f} MFLOPs)  "
          f"| thop: MACs {thop_macs / 1e6:.2f} M, params {int(thop_params):,} (thop counts BN/ReLU ops too)")

    # Size / compute of the standard models at the same input size, for reference.
    summary = {tag: {"params": total_p, "macs": total_m}}
    for name in ("resnet18", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"):
        m = build_model(name, in_hw)
        summary[name] = {"params": count_params(m), "macs": model_macs(m, in_hw)}
    (out / f"compare_{args.size}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nat {args.size}:")
    for k, v in summary.items():
        print(f"  {k:28s} params {v['params']:>11,}  MACs {v['macs'] / 1e6:8.1f} M")


if __name__ == "__main__":
    main()
