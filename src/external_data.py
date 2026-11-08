"""Real-microscopy external test sets: counts and metadata.

Used only for testing, never for training or tuning.

- BBBC039 (U2OS, Hoechst, 520x696, 12-bit in 16-bit TIFF, CC0): exact counts
  from the instance masks. Mask PNG red channel holds a small label (1..k);
  touching nuclei get different labels, so a nucleus = one connected component
  of one label value. Components smaller than MIN_AREA px are dropped as
  annotation fragments (the effect is reported).
- BBBC002 (Drosophila Kc167, Hoechst, 512x512, 8-bit): count = mean of two
  human counters.
- BBBC001 (human HT29, Hoechst, 512x512, 8-bit, CC BY-NC-SA 3.0): count = mean
  of two human counters.

Writes data/external/<name>.csv (one row per image) and prints summaries.
Usage: python -m src.external_data
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "external"
MIN_AREA = 10
EIGHT = np.ones((3, 3), dtype=int)


def bbbc039_counts():
    zm = zipfile.ZipFile(RAW / "BBBC039" / "masks.zip")
    zmeta = zipfile.ZipFile(RAW / "BBBC039" / "metadata.zip")
    subset = {}
    for part in ("training", "validation", "test"):
        for line in zmeta.read(f"metadata/{part}.txt").decode().split():
            subset[Path(line).stem] = part
    rows = []
    for name in sorted(n for n in zm.namelist() if n.startswith("masks/") and n.endswith(".png")):
        lab = np.asarray(Image.open(io.BytesIO(zm.read(name))))[..., 0]
        areas = []
        for v in np.unique(lab[lab > 0]):
            comp, n = ndimage.label(lab == v, structure=EIGHT)
            areas += list(np.bincount(comp.ravel())[1:])
        areas = np.array(areas)
        stem = Path(name).stem
        rows.append({"image": f"images/{stem}.tif", "mask": name,
                     "count": int((areas >= MIN_AREA).sum()),
                     "count_all_components": int(areas.size),
                     "median_area_px": float(np.median(areas[areas >= MIN_AREA])) if areas.size else 0.0,
                     "bbbc039_subset": subset.get(stem, "?")})
    return pd.DataFrame(rows)


def human_counts(name, counts_file, zip_file):
    df = pd.read_csv(RAW / name / counts_file, sep="\t")
    df.columns = ["file", "counter1", "counter2"]
    members = {Path(m).stem.lower(): m for m in zipfile.ZipFile(RAW / name / zip_file).namelist()
               if m.lower().endswith((".tif", ".tiff")) and "__MACOSX" not in m}
    df["image"] = df["file"].map(lambda f: members.get(Path(f).stem.lower()))
    missing = df["image"].isna().sum()
    assert missing == 0, f"{name}: {missing} count rows without an image"
    df["count"] = (df["counter1"] + df["counter2"]) / 2
    df["counter_rel_diff"] = (df["counter1"] - df["counter2"]).abs() / df["count"]
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d39 = bbbc039_counts()
    d39.to_csv(OUT / "bbbc039.csv", index=False)
    d02 = human_counts("BBBC002", "BBBC002_v1_counts.txt", "BBBC002_v1_images.zip")
    d02["sample"] = d02["file"].str.extract(r"CPvalid1_([^_]+)_")[0]
    d02.to_csv(OUT / "bbbc002.csv", index=False)
    d01 = human_counts("BBBC001", "BBBC001_v1_counts.txt", "BBBC001_v1_images_tif.zip")
    d01.to_csv(OUT / "bbbc001.csv", index=False)

    c = d39["count"]
    print(f"BBBC039: {len(d39)} images, {c.sum():,} nuclei (all components "
          f"{d39.count_all_components.sum():,}); per image min {c.min()} median {c.median():.0f} "
          f"max {c.max()}; >100: {(c > 100).sum()}; median nucleus area "
          f"{d39.median_area_px.median():.0f} px; subsets {d39.bbbc039_subset.value_counts().to_dict()}")
    for name, d in (("BBBC002", d02), ("BBBC001", d01)):
        c = d["count"]
        print(f"{name}: {len(d)} images; count min {c.min()} median {c.median()} max {c.max()}; "
              f"mean human disagreement {100 * d.counter_rel_diff.mean():.1f}%")


if __name__ == "__main__":
    main()
