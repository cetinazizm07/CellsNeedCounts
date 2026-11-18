"""External test on real microscopy (BBBC039, BBBC002, BBBC001).

Test only: nothing here is trained or tuned on these images, and every rule
below was fixed before any model prediction on them was looked at. Models are
the final checkpoints used in results/results.csv.

Domain gaps to BBBC005 (synthetic, 520x696 8-bit) and how each is handled by a
fixed, label-free rule:

1. Intensity. Real images are 12-bit (BBBC039) or dim 8-bit. Per image, a
   linear map sends its 1st and 99.5th percentiles to the medians of the same
   percentiles over the BBBC005 train images at the model's input resolution;
   values are then clipped to 0..255 and rounded to uint8 (as in training).
2. Nucleus size. Estimated identically on every dataset, without labels: Otsu
   threshold, 4-connected components >= 20 px, median component area at the
   original resolution (BBBC005 reference: train images of the sharpest blur
   level). Diameter ~ sqrt(area). For BBBC039 the estimate is checked against
   the median mask area.
3. Count range. BBBC005 trains on 1..100 cells; most BBBC039 images have more.

Protocols (all reported, image-level metrics on the true per-image count):
  naive          whole image resized to the model's training input size,
                 ignoring nucleus size (what a user would do first)
  matched        whole image resized so nuclei have the BBBC005 size at the
                 model's input resolution; fully convolutional own model and
                 adaptive-pool standard models both accept the larger input
  tiles          (BBBC039 only) the matched image, snapped to k x k tiles of
                 exactly the training input size; tile truth = mask nuclei
                 whose centroid falls in the tile; image prediction = sum of
                 tile predictions. Tile-level metrics are reported too.
BBBC002/BBBC001 have human counts only (mean of two counters), so no tiles.

Energy: preparation is one measured run (ext_prep), each model one measured
run (ext_<run>), all appended to logs/runs.csv.

Usage: python -m src.external_eval [--force]
       -> results/external/{calibration.json, <run>.json, <run>_predictions.csv, summary.csv}
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage

from src.energy import measure_run
from src.evaluate import load_trained
from src.external_data import EIGHT, MIN_AREA
from src.metrics import regression_metrics
from src.train import CACHE, predict

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
EXT = ROOT / "data" / "external"
OUT = ROOT / "results" / "external"
RUNS = ["own_green_48x64", "own_e15", "mnv3s_ft_e15", "resnet18_ft_e15",
        "mnv3s_scratch_e15", "resnet18_probe_e15",
        # controlled sum-vs-average test: identical to own_e15 except the pooling head
        "own_abl_pool_avg"]
P_LO, P_HI = 1.0, 99.5
MIN_COMP = 20  # px at original resolution, for the size estimate only
BBBC005_H = 520
N_REF = 200  # BBBC005 train images used for the nucleus size reference


def otsu(x, bins=256):
    lo, hi = np.percentile(x, [0.1, 99.9])
    h, edges = np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi))
    p = h / h.sum()
    w0 = np.cumsum(p)
    mu = np.cumsum(p * np.arange(bins))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0))
    return edges[np.nanargmax(between) + 1]


def component_areas(img):
    comp, n = ndimage.label(img > otsu(img))
    a = np.bincount(comp.ravel())[1:]
    return a[a >= MIN_COMP]


def read_tif(zf, member):
    with Image.open(io.BytesIO(zf.read(member))) as im:
        if im.mode == "P":
            im = im.convert("L")
        return np.asarray(im).astype(np.float32)


def load_raw():
    sets = {}
    zf = zipfile.ZipFile(RAW / "BBBC039" / "images.zip")
    zm = zipfile.ZipFile(RAW / "BBBC039" / "masks.zip")
    d = pd.read_csv(EXT / "bbbc039.csv")
    imgs, cents = [], []
    for r in d.itertuples():
        imgs.append(read_tif(zf, r.image))
        lab = np.asarray(Image.open(io.BytesIO(zm.read(r.mask))))[..., 0]
        c = []
        for v in np.unique(lab[lab > 0]):  # same nucleus definition as external_data.py
            comp, n = ndimage.label(lab == v, structure=EIGHT)
            sizes = np.bincount(comp.ravel())[1:]
            keep = [i + 1 for i in range(n) if sizes[i] >= MIN_AREA]
            c += ndimage.center_of_mass(np.ones_like(comp), comp, keep)
        assert len(c) == r.count, r.image
        cents.append(np.array(c).reshape(-1, 2))
    sets["BBBC039"] = {"df": d, "imgs": imgs, "centroids": cents}
    for name, zname in (("BBBC002", "BBBC002_v1_images.zip"), ("BBBC001", "BBBC001_v1_images_tif.zip")):
        d = pd.read_csv(EXT / f"{name.lower()}.csv")
        zf = zipfile.ZipFile(RAW / name / zname)
        sets[name] = {"df": d, "imgs": [read_tif(zf, m) for m in d.image]}
    return sets


def bbbc005_reference():
    meta = pd.read_csv(CACHE / "meta_w2.csv")
    sharp = meta[(meta.split == "train") & (meta.blur == meta.blur.min())]
    sharp = sharp.sample(min(N_REF, len(sharp)), random_state=655)
    zf = zipfile.ZipFile(RAW / "BBBC005" / "BBBC005_v1_images.zip")
    names = {Path(n).name: n for n in zf.namelist()}
    areas = np.concatenate([component_areas(read_tif(zf, names[f])) for f in sharp.filename])
    ref = {"median_area_px": float(np.median(areas)), "n_images": len(sharp)}
    arr = np.load(CACHE / "images_w2_96x128.npy", mmap_mode="r")
    tr = np.asarray(arr[meta.index[meta.split == "train"].to_numpy()]).astype(np.float32)
    for hw, x in (("96x128", tr), ("48x64", tr.reshape(-1, 48, 2, 64, 2).mean(axis=(2, 4)).round())):
        lo, hi = np.percentile(x.reshape(len(x), -1), [P_LO, P_HI], axis=1)
        ref[hw] = {"p_lo": float(np.median(lo)), "p_hi": float(np.median(hi))}
    return ref


def resize(img, h, w):
    return np.asarray(Image.fromarray(img.astype(np.float32)).resize((w, h), Image.BOX))


def match_intensity(x, ref):
    lo, hi = np.percentile(x, [P_LO, P_HI])
    y = (x - lo) / max(hi - lo, 1e-6) * (ref["p_hi"] - ref["p_lo"]) + ref["p_lo"]
    return np.clip(y, 0, 255).round().astype(np.uint8)


def build_inputs(sets, cal, in_hw):
    """Return {(dataset, protocol): (uint8 array N x H x W, truth, extra)}."""
    hm, wm = in_hw
    ref = cal["bbbc005"][f"{hm}x{wm}"]
    d005 = np.sqrt(cal["bbbc005"]["median_area_px"]) * hm / BBBC005_H  # nucleus diam at model input
    out = {}
    for name, s in sets.items():
        H, W = s["imgs"][0].shape
        r = d005 / np.sqrt(cal[name]["median_area_px"])
        y = s["df"]["count"].to_numpy(dtype=np.float64)
        out[(name, "naive")] = (np.stack([match_intensity(resize(i, hm, wm), ref) for i in s["imgs"]]), y, None)
        if name == "BBBC039":
            k = max(1, int(round(H * r / hm)))
            mh, mw = k * hm, k * wm
            cal[name][f"tiles_k_{hm}x{wm}"] = k
        else:
            mh, mw = max(8, int(round(H * r / 8)) * 8), max(8, int(round(W * r / 8)) * 8)
        cal[name][f"matched_size_{hm}x{wm}"] = [mh, mw]
        cal[name][f"scale_error_{hm}x{wm}"] = float(mh / (H * r) - 1)  # from snapping
        full = np.stack([match_intensity(resize(i, mh, mw), ref) for i in s["imgs"]])
        out[(name, "matched")] = (full, y, None)
        if name == "BBBC039":
            tiles, ty, owner = [], [], []
            for n, (img, c) in enumerate(zip(full, s["centroids"])):
                ti = np.minimum((c[:, 0] * k / H).astype(int), k - 1)
                tj = np.minimum((c[:, 1] * k / W).astype(int), k - 1)
                for i in range(k):
                    for j in range(k):
                        tiles.append(img[i * hm:(i + 1) * hm, j * wm:(j + 1) * wm])
                        ty.append(int(((ti == i) & (tj == j)).sum()))
                        owner.append(n)
            out[(name, "tiles")] = (np.stack(tiles), y, (np.array(ty, float), np.array(owner)))
    return out


def run_model(run, sets, cal, dev):
    model, a, in_hw = load_trained(run, dev)
    inputs = build_inputs(sets, cal, in_hw)
    res, preds = {"run": run, "in_hw": list(in_hw), "results": {}}, []
    for (name, proto), (x, y, extra) in inputs.items():
        p = predict(model, torch.from_numpy(x).unsqueeze(1).to(dev)).cpu().numpy().astype(np.float64)
        entry = {}
        if extra is not None:
            ty, owner = extra
            entry["tile"] = regression_metrics(ty, p)
            entry["tile"]["truth_range"] = [float(ty.min()), float(ty.max())]
            p = np.bincount(owner, weights=p, minlength=len(y))
        m = regression_metrics(y, p)
        m["rel_mae"] = m["mae"] / float(np.mean(y))
        m["mean_count"] = float(np.mean(y))
        entry["image"] = m
        res["results"][f"{name}/{proto}"] = entry
        preds.append(pd.DataFrame({"dataset": name, "protocol": proto, "image": sets[name]["df"].image,
                                   "count": y, "pred": p}))
    return res, pd.concat(preds, ignore_index=True)


LABELS = {"own_green_48x64": "Own CNN final (48x64, sum)", "own_e15": "Own CNN reference (96x128, sum)",
          "own_abl_pool_avg": "Own CNN reference, avg pool", "mnv3s_ft_e15": "MobileNetV3-S FT",
          "resnet18_ft_e15": "ResNet-18 FT", "mnv3s_scratch_e15": "MobileNetV3-S scratch",
          "resnet18_probe_e15": "ResNet-18 frozen"}
COLS = ["BBBC039/naive", "BBBC039/matched", "BBBC039/tiles", "BBBC002/naive", "BBBC002/matched",
        "BBBC001/naive", "BBBC001/matched"]


def write_table():
    """Relative MAE (MAE / mean true count) per model and set; image level."""
    df = pd.read_csv(OUT / "summary.csv")
    piv = df.pivot(index="run", columns="set", values="img_rel_mae").reindex(
        [r for r in LABELS if r in set(df.run)])[COLS]
    bias = df.pivot(index="run", columns="set", values="img_bias")
    means = df.drop_duplicates("set").set_index("set").img_mean_count
    lines = ["| Model | " + " | ".join(COLS) + " |", "|---|" + "---:|" * len(COLS),
             "| *mean true count* | " + " | ".join(f"*{means[c]:.0f}*" for c in COLS) + " |"]
    for run, r in piv.iterrows():
        cells = [f"**{v:.3f}**" if v == piv[c].min() else f"{v:.3f}" for c, v in r.items()]
        lines.append(f"| {LABELS[run]} | " + " | ".join(cells) + " |")
    lines += ["", "Signed bias (cells, + = over-count):", "",
              "| Model | " + " | ".join(COLS) + " |", "|---|" + "---:|" * len(COLS)]
    for run in piv.index:
        lines.append(f"| {LABELS[run]} | " + " | ".join(f"{bias.loc[run, c]:+.1f}" for c in COLS) + " |")
    md = "\n".join(lines) + "\n"
    (OUT / "table.md").write_text(md)
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--prep-only", action="store_true", help="calibration only, no model predictions")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    done = [] if args.prep_only else [r for r in args.runs if (OUT / f"{r}.json").exists()]
    if done and not args.force:
        raise SystemExit(f"external results already exist for {done}; use --force to overwrite")
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    with measure_run("ext_prep", phase="preprocess", model="-",
                     notes="load BBBC039/002/001, mask centroids, label-free size/intensity calibration"):
        sets = load_raw()
        cal = {"bbbc005": bbbc005_reference()}
        for name, s in sets.items():
            areas = np.concatenate([component_areas(i) for i in s["imgs"]])
            cal[name] = {"median_area_px": float(np.median(areas)), "n_images": len(s["imgs"]),
                         "orig_hw": list(s["imgs"][0].shape)}
        cal["BBBC039"]["mask_median_area_px"] = float(sets["BBBC039"]["df"].median_area_px.median())
        for name in sets:
            cal[name]["diameter_ratio_to_bbbc005"] = float(
                np.sqrt(cal[name]["median_area_px"] / cal["bbbc005"]["median_area_px"]))
    print(json.dumps(cal, indent=1))
    if args.prep_only:
        return

    rows = []
    for run in args.runs:
        with measure_run(f"ext_{run}", phase="external", model=run, notes="real-microscopy external test"):
            res, preds = run_model(run, sets, cal, dev)
        (OUT / f"{run}.json").write_text(json.dumps(res, indent=2))
        preds.to_csv(OUT / f"{run}_predictions.csv", index=False)
        for key, e in res["results"].items():
            rows.append({"run": run, "set": key, **{f"img_{k}": v for k, v in e["image"].items()},
                         **({f"tile_{k}": v for k, v in e["tile"].items() if k != "truth_range"}
                            if "tile" in e else {})})
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=2))
    df = pd.DataFrame(rows)
    old = OUT / "summary.csv"
    if old.exists():  # keep rows of runs evaluated in an earlier call
        prev = pd.read_csv(old)
        df = pd.concat([prev[~prev.run.isin(args.runs)], df], ignore_index=True)
    df.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    write_table()
    print(df[["run", "set", "img_mae", "img_rel_mae", "img_bias", "img_r2", "tile_mae"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
