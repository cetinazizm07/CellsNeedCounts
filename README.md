# CSE 655 Term Project — Green CNNs for Image Regression

Status: environment set up; dataset and code not added yet.

## Hardware

- Apple M2 (8-core GPU, 8 GB unified memory), macOS, local training
- Backend: PyTorch with the MPS (Metal) device
- Location for CO2 intensity: Türkiye (`country_iso_code="TUR"`)

## Environment

```bash
conda create -n cse655 python=3.11
conda activate cse655
pip install -r requirements.txt
```

Sanity check (2026-11-08): ResNet-18, 128×128 input, batch 64, AdamW on MPS
runs at about 210 ms/step (~300 img/s).

## Energy measurement

Every run (training, evaluation, searches, failed runs) is wrapped in
`src.energy.measure_run`, which appends one row to `logs/runs.csv`:

```python
from src.energy import measure_run
with measure_run("owncnn_run07", phase="train", model="own_cnn"):
    train(...)
```

Two measurements are taken side by side:

| | Source | Output |
|---|---|---|
| Primary | `src/power_meter.py`: continuous `powermetrics` (CPU+GPU+ANE), 1 s samples, energy = Σ power × elapsed | `logs/power/<run>_<time>.csv`, `meter_*` columns |
| Reference | CodeCarbon 3.3.2 `OfflineEmissionsTracker` (required by the handout) | `logs/codecarbon/emissions.csv`, `cc_*` columns |
| Raw CodeCarbon input | every powermetrics window CodeCarbon read (it overwrites `powermetrics_log.txt` each time) | `logs/codecarbon/raw/<run>_<time>.txt.gz` (`gzip -dc` to read) |

CO2 = kWh × 0.464586 kg/kWh, the Türkiye grid intensity CodeCarbon uses
(`global_energy_mix.json`, 2023).

### One-time setup: passwordless powermetrics

`powermetrics` needs root. `setup/codecarbon_powermetrics.sudoers` allows it
without a password **only with the exact arguments** CodeCarbon and the meter
use (a blanket rule would let any process write root-owned files via `-o`).

```bash
sudo install -m 0440 -o root -g wheel setup/codecarbon_powermetrics.sudoers /etc/sudoers.d/codecarbon_powermetrics && sudo visudo -c
python setup/check_energy.py   # expect "GPU energy measured: OK"
```

Remove with `sudo rm /etc/sudoers.d/codecarbon_powermetrics`.

### Issues found while setting this up (2026-11-08)

1. The handout's snippet `EmissionsTracker(country_iso_code="TUR")` raises
   `TypeError` in CodeCarbon 3.3.2; `country_iso_code` is only accepted by
   `OfflineEmissionsTracker`.
2. Without sudo, CodeCarbon on Apple Silicon falls back to "CPU load mode" and
   reports **GPU power = 0 W**.
3. Even with sudo, CodeCarbon 3.3.2 never reaches powermetrics on Apple
   Silicon: a "CPU load" fast path is tried first and always succeeds when
   psutil is installed. `src/energy.py` disables that fast path when
   powermetrics is usable.
4. With powermetrics active, CodeCarbon still under-counts GPU energy: it
   samples ~1 s per measurement and applies it to the whole interval, and the
   last sample is taken after training stops. Steady ResNet-18 load on the M2:
   raw powermetrics GPU 7–9 W; CodeCarbon average 2.4–5.1 W depending on
   `measure_power_secs`. Hence the continuous meter as the primary number.

Check run (ResNet-18, 128 px, batch 64, 30 s of steps): meter average GPU
6.71 W, CPU 2.68 W, sample coverage 98.7 % of the run.

Limitations: the meter covers the SoC (CPU, GPU, ANE) only — not DRAM,
display, SSD or charger losses. CodeCarbon's numbers additionally include a
RAM power estimate.

## Dataset: BBBC005 v1 (synthetic cells)

- Source: <https://bbbc.broadinstitute.org/BBBC005>. Copyright waived (CC0-style).
  Cite: "We used the image set BBBC005v1 from the Broad Bioimage Benchmark
  Collection [Ljosa et al., Nature Methods, 2012]."
- `BBBC005_v1_images.zip`: 1,882,973,059 bytes,
  SHA-256 `9ab5e7e1d544f5b88f43b6fc8c2236ff63a0659c9ba17b850d43f290d80c6815`.
- 19,200 TIFs, 696×520, 8-bit gray. Name: `SIMCEPImages_<well>_C<count>_F<blur>_s<sample>_w<stain>.TIF`.
  24 count values (1…100), 16 blur levels (F1…F48), 25 samples, 2 stains.
- We use stain w2 (nuclei): **9,600 images**. Target = `count`.
- **Layout reuse:** the same synthetic layout (count, sample) is rendered at
  all 16 blur levels (pixel correlation F1 vs F4 = 0.998; different samples
  0.021). The split is therefore per layout, stratified by count, seed 655:
  17 / 4 / 4 layouts per count → train 6,528 images (408 layouts),
  val 1,536 (96), test 1,536 (96). No layout appears in two splits; every
  split contains all 24 counts and all 16 blur levels.
- Caveat: only 24 distinct count values exist and all appear in every split,
  so the test measures accuracy on seen count values, not interpolation.

## External test sets: real microscopy (test only, never used for training or tuning)

| Set | Cells / stain | Size | Images | Count label | Licence | SHA-256 (zip) |
|---|---|---|---:|---|---|---|
| [BBBC039](https://bbbc.broadinstitute.org/BBBC039) | human U2OS, Hoechst | 520×696, 12-bit in 16-bit TIFF | 200 | exact, from instance masks | CC0 | images `6f30a5d4…ddd7`, masks `f9e6043d…2710` |
| [BBBC002](https://bbbc.broadinstitute.org/BBBC002) | Drosophila Kc167, Hoechst | 512×512, 8-bit | 50 | mean of 2 human counters | not stated on page | `19d74948…009f` |
| [BBBC001](https://bbbc.broadinstitute.org/BBBC001) | human HT29, Hoechst | 512×512, 8-bit | 6 | mean of 2 human counters | CC BY-NC-SA 3.0 | `d6d4ad47…d48` |

Full checksums: `shasum -a 256 data/raw/BBBC0*/*`. Citations: BBBC039v1 Caicedo et al. 2018;
BBBC001v1 / BBBC002v1 Carpenter et al., Genome Biology 2006; all via Ljosa et al., Nature Methods 2012.

`python -m src.external_data` writes `data/external/{bbbc039,bbbc002,bbbc001}.csv`:

| Set | Count min / median / max | > 100 | Nucleus diameter* | Human counters disagree |
|---|---|---:|---|---:|
| BBBC005 (training data) | 1 / 50.5 / 100 | 0 | ~64 px at 696×520 | – |
| BBBC039 | 0 / 124 / 230 (total 23,556) | 140 of 200 | ~28 px at 696×520 | – |
| BBBC002 | 8 / 41.5 / 185 | some | ~19 px at 512×512 | 19.7 % |
| BBBC001 | 249 / 347.5 / 465.5 | all | ~11 px at 512×512 | 9.8 % |

\* Median area of Otsu-thresholded components (BBBC039: mask components), as an
equivalent-circle diameter. BBBC039 mask decoding: red channel = label 1..k,
touching nuclei get different labels, so nucleus = connected component (8-conn.)
of one label; components < 10 px are dropped (23,615 → 23,556).

Consequences for testing: the real nuclei are 2–6× smaller in pixels than the
synthetic ones, and most real images hold more than 100 nuclei. Images must be
rescaled so nuclei have the training size (~11 px at the model input), and
counts above 100 test extrapolation.

## Reproducing results

```bash
mkdir -p data/raw/BBBC005
curl -L -C - -o data/raw/BBBC005/BBBC005_v1_images.zip https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip
python -m src.data_prep    # -> data/cache/images_w2_{96x128,192x256}.npy, meta_w2.csv
python -m src.baseline     # -> results/baseline.json
```

| Step | Result |
|---|---|
| Preprocessing (box downsampling, both sizes) | 28.3 s, 4.5e-5 kWh |
| Mean / median baseline (both 50.5) | test MAE 25.833, RMSE 29.773, R² 0.000 |

Acceptability thresholds (handout §3.5): MAE_own ≤ 0.6 × 25.833 = **15.50**,
and MAE_own ≤ 1.15 × MAE_best (best standard model, still to be trained).

To be filled in: training commands per model, evaluation, `results.csv`, error–CO2 plot.
