"""Single place where every run's energy and CO2 are measured.

Use it for every training / evaluation / search run, including failed ones:

    from src.energy import measure_run
    with measure_run("owncnn_run07", phase="train", model="own_cnn") as m:
        train(...)
    # m.row has the numbers; a row is also appended to logs/runs.csv

Two measurements are taken side by side:

1. Primary: `src/power_meter.PowerMeter` integrates continuous powermetrics
   samples (CPU + GPU + ANE) over the whole run.
2. Reference: CodeCarbon (OfflineEmissionsTracker), which writes
   logs/codecarbon/emissions.csv as required by the handout. Its Apple GPU
   numbers under-count (see power_meter.py), so it is kept for completeness,
   not as the headline number.

CO2 for the primary number uses the same grid intensity CodeCarbon uses for
Türkiye (global_energy_mix.json, 2023): 464.586 gCO2eq/kWh.

Patch for CodeCarbon 3.3.2
--------------------------
On Apple Silicon it tries a "CPU load" fast path *before* powermetrics
(core/resource_tracker.py, `_try_platform_cpu_backend`). The fast path
succeeds whenever psutil is installed, so powermetrics is never reached and
GPU power is reported as 0 W, even with passwordless sudo. (3.2.8 and earlier
lack the fast path but call powermetrics with an extra empty argument that
does not match our restricted sudoers rule.) We disable the fast path only
when powermetrics is usable.
"""
import contextlib
import csv
import gzip
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from codecarbon import OfflineEmissionsTracker
from codecarbon.core import powermetrics
from codecarbon.core.powermetrics import ApplePowermetrics
from codecarbon.core.resource_tracker import ResourceTracker

from src.power_meter import PowerMeter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Must match the -o path pinned in setup/codecarbon_powermetrics.sudoers
CC_LOG_DIR = PROJECT_ROOT / "logs" / "codecarbon"
CC_RAW_DIR = CC_LOG_DIR / "raw"
POWER_LOG_DIR = PROJECT_ROOT / "logs" / "power"
RUNS_CSV = PROJECT_ROOT / "logs" / "runs.csv"
COUNTRY = "TUR"
GRID_KG_PER_KWH = 0.464586
# CodeCarbon samples powermetrics for ~1 s at each measurement and applies that
# power to the whole interval since the previous one.
MEASURE_POWER_SECS = 15

_original_fast_path = ResourceTracker._setup_cpu_load_fast


def _fast_path_unless_powermetrics(self, model):
    if powermetrics.is_powermetrics_available():
        return False
    return _original_fast_path(self, model)


ResourceTracker._setup_cpu_load_fast = _fast_path_unless_powermetrics

# CodeCarbon overwrites logs/codecarbon/powermetrics_log.txt at every
# measurement. While a run is active, each fresh copy is appended to
# logs/codecarbon/raw/<run>_<stamp>.txt.gz (one gzip member per measurement,
# readable with `gzip -dc` or `zcat`), so the raw CodeCarbon input is kept.
_raw_archive_path = None
_original_log_values = ApplePowermetrics._log_values


def _log_values_and_archive(self) -> bool:
    ok = _original_log_values(self)
    if ok and _raw_archive_path is not None:
        try:
            text = Path(self._log_file_path).read_text()
        except OSError:
            return ok
        stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with gzip.open(_raw_archive_path, "at") as f:
            f.write(f"##### codecarbon powermetrics measurement {stamp}\n{text}\n")
    return ok


ApplePowermetrics._log_values = _log_values_and_archive


def start_tracker(run_name: str, save_to_file: bool = True,
                  measure_power_secs: float = MEASURE_POWER_SECS) -> OfflineEmissionsTracker:
    """Create and start a CodeCarbon tracker writing to logs/codecarbon/.

    Raises if CodeCarbon did not pick powermetrics, so no run is silently
    measured in a different mode than the others.
    """
    CC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not powermetrics.is_powermetrics_available():
        raise RuntimeError(
            "powermetrics is not usable without a password; GPU energy would be "
            "reported as 0 W. Install setup/codecarbon_powermetrics.sudoers (see README)."
        )
    tracker = OfflineEmissionsTracker(
        project_name=run_name,
        country_iso_code=COUNTRY,
        output_dir=str(CC_LOG_DIR),
        save_to_file=save_to_file,
        measure_power_secs=measure_power_secs,
        log_level="warning",
    )
    tracker.start()  # hardware backends are chosen here, not in __init__
    hw = [type(h).__name__ for h in tracker._hardware]
    if "AppleSiliconChip" not in hw:
        tracker.stop()
        raise RuntimeError(f"CodeCarbon did not select powermetrics: {hw}")
    return tracker


class RunMeasurement:
    row: dict = None


RUN_FIELDS = [
    "run_name", "phase", "model", "status", "start_utc", "duration_s",
    "meter_cpu_kwh", "meter_gpu_kwh", "meter_ane_kwh", "meter_total_kwh",
    "meter_kg_co2eq", "meter_coverage", "cc_energy_kwh", "cc_kg_co2eq",
    "hardware", "notes",
]


@contextlib.contextmanager
def measure_run(run_name: str, phase: str, model: str, notes: str = "",
                save: bool = True):
    """Measure a run with the power meter and CodeCarbon; log it to runs.csv.

    The row is written even if the body raises (status="failed"), because
    failed runs count towards the project footprint.
    """
    global _raw_archive_path
    m = RunMeasurement()
    start_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if save:
        CC_RAW_DIR.mkdir(parents=True, exist_ok=True)
        _raw_archive_path = CC_RAW_DIR / f"{run_name}_{stamp}.txt.gz"
    meter = PowerMeter().start()
    tracker = start_tracker(run_name, save_to_file=save)
    status = "ok"
    try:
        yield m
    except BaseException:
        status = "failed"
        raise
    finally:
        tracker.stop()
        _raw_archive_path = None
        res = meter.stop()
        cc = tracker.final_emissions_data
        kwh = lambda j: j / 3.6e6  # noqa: E731
        m.row = {
            "run_name": run_name, "phase": phase, "model": model,
            "status": status, "start_utc": start_utc,
            "duration_s": round(res.duration_s, 2),
            "meter_cpu_kwh": kwh(res.cpu_j), "meter_gpu_kwh": kwh(res.gpu_j),
            "meter_ane_kwh": kwh(res.ane_j), "meter_total_kwh": res.total_kwh,
            "meter_kg_co2eq": res.total_kwh * GRID_KG_PER_KWH,
            "meter_coverage": round(res.covered_s / res.duration_s, 3) if res.duration_s else 0,
            "cc_energy_kwh": cc.energy_consumed, "cc_kg_co2eq": cc.emissions,
            "hardware": f"{cc.cpu_model} / {platform.platform()}",
            "notes": notes,
        }
        if save:
            res.write_csv(POWER_LOG_DIR / f"{run_name}_{stamp}.csv")
            RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
            new = not RUNS_CSV.exists()
            with open(RUNS_CSV, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(m.row)
