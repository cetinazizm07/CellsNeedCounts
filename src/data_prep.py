"""Build the cached BBBC005 arrays and the fixed train/val/test split.

Reads TIFs straight from the downloaded zip (no extraction to disk), keeps one
stain, downsamples with box filtering and saves uint8 arrays.

Dataset facts (BBBC005 v1, https://bbbc.broadinstitute.org/BBBC005):
  SIMCEPImages_<well>_C<cells>_F<blur>_s<sample>_w<stain>.TIF, 696x520, 8-bit.
  The target is <cells>. The same synthetic layout (cells, sample) is rendered
  at every blur level and in both stains, so the split is done per layout
  (group split) to avoid leaking a layout into val/test.

Usage:
  python -m src.data_prep               # defaults below
"""
import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.energy import measure_run

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data" / "raw" / "BBBC005" / "BBBC005_v1_images.zip"
OUT = ROOT / "data" / "cache"
SEED = 655
NAME_RE = re.compile(
    r"SIMCEPImages_(?P<well>[A-P]\d{2})_C(?P<count>\d+)_F(?P<blur>\d+)"
    r"_s(?P<sample>\d+)_w(?P<stain>\d)\.TIF$", re.IGNORECASE)


def parse_names(zf: zipfile.ZipFile) -> pd.DataFrame:
    rows = []
    for name in zf.namelist():
        m = NAME_RE.search(name)
        if m:
            d = m.groupdict()
            rows.append({"member": name, "well": d["well"], "count": int(d["count"]),
                         "blur": int(d["blur"]), "sample": int(d["sample"]),
                         "stain": int(d["stain"])})
    df = pd.DataFrame(rows).sort_values(["stain", "count", "sample", "blur"])
    return df.reset_index(drop=True)


def group_split(df: pd.DataFrame, val_per_count: int, test_per_count: int,
                seed: int) -> pd.Series:
    """Assign each (count, sample) layout to one split, stratified by count."""
    rng = np.random.default_rng(seed)
    split = {}
    for count, g in df.groupby("count"):
        samples = np.sort(g["sample"].unique())
        rng.shuffle(samples)
        for i, s in enumerate(samples):
            split[(count, s)] = ("test" if i < test_per_count else
                                 "val" if i < test_per_count + val_per_count else "train")
    return df.apply(lambda r: split[(r["count"], r["sample"])], axis=1)


def load_resized(zf, member, sizes):
    with Image.open(io.BytesIO(zf.read(member))) as im:
        im = im.convert("L")
        return [np.asarray(im.resize((w, h), Image.BOX), dtype=np.uint8)
                for (h, w) in sizes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stain", type=int, default=2, help="1=cell body, 2=nuclei")
    ap.add_argument("--sizes", default="96x128,192x256", help="HxW list")
    ap.add_argument("--val-per-count", type=int, default=4)
    ap.add_argument("--test-per-count", type=int, default=4)
    args = ap.parse_args()
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]

    OUT.mkdir(parents=True, exist_ok=True)
    with measure_run("data_prep", phase="preprocess", model="-",
                     notes=f"stain={args.stain} sizes={args.sizes}"):
        with zipfile.ZipFile(ZIP) as zf:
            df = parse_names(zf)
            df = df[df["stain"] == args.stain].reset_index(drop=True)
            df["layout_id"] = df["count"].astype(str) + "_" + df["sample"].astype(str)
            df["split"] = group_split(df, args.val_per_count, args.test_per_count, SEED)
            arrays = [np.empty((len(df), h, w), dtype=np.uint8) for (h, w) in sizes]
            for i, member in enumerate(df["member"]):
                for arr, img in zip(arrays, load_resized(zf, member, sizes)):
                    arr[i] = img
                if (i + 1) % 1000 == 0:
                    print(f"{i + 1}/{len(df)}")
        for (h, w), arr in zip(sizes, arrays):
            np.save(OUT / f"images_w{args.stain}_{h}x{w}.npy", arr)
        df.drop(columns="member").assign(filename=df["member"].map(lambda p: Path(p).name)) \
          .to_csv(OUT / f"meta_w{args.stain}.csv", index_label="idx")

    print(df.groupby("split").agg(images=("count", "size"),
                                  layouts=("layout_id", "nunique")))


if __name__ == "__main__":
    main()
