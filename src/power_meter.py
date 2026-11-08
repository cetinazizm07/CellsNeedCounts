"""Continuous Apple Silicon power meter based on `powermetrics`.

Why not CodeCarbon alone: CodeCarbon 3.3.2 takes a ~1 s powermetrics sample
at each measurement point and applies it to the whole interval since the
previous point; the final sample is taken after training has stopped. In a
check with a steady ResNet-18 load (raw powermetrics: GPU 7-9 W), CodeCarbon
reported an average GPU power of only 2.4-5.1 W depending on the interval.

This meter runs `powermetrics --samplers cpu_power -i 1000` for the whole run.
Each sample is the average power over its own elapsed window, so
energy = sum(power_i * elapsed_i) with no gaps (except the final partial
second when the run stops). It reads powermetrics' stdout, so the root process
never writes a file.

Scope: CPU + GPU + ANE power of the SoC as reported by powermetrics. DRAM,
display, SSD and power-supply losses are not included.
"""
import csv
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Must match setup/codecarbon_powermetrics.sudoers exactly.
CMD = ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "cpu_power", "-i", "1000"]

_HEADER = re.compile(r"\*\*\* Sampled system activity .*\(([\d.]+)ms elapsed\) \*\*\*")
_POWER = re.compile(r"^(CPU|GPU|ANE) Power: (\d+) mW")


@dataclass
class PowerSample:
    t_wall: float          # time.time() when the sample was parsed
    elapsed_s: float       # window covered by this sample
    cpu_w: float = 0.0
    gpu_w: float = 0.0
    ane_w: float = 0.0


@dataclass
class PowerResult:
    duration_s: float
    samples: list = field(default_factory=list)

    def _joules(self, attr):
        return sum(getattr(s, attr) * s.elapsed_s for s in self.samples)

    @property
    def cpu_j(self): return self._joules("cpu_w")
    @property
    def gpu_j(self): return self._joules("gpu_w")
    @property
    def ane_j(self): return self._joules("ane_w")
    @property
    def total_j(self): return self.cpu_j + self.gpu_j + self.ane_j
    @property
    def total_kwh(self): return self.total_j / 3.6e6
    @property
    def covered_s(self): return sum(s.elapsed_s for s in self.samples)

    def write_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_wall", "elapsed_s", "cpu_w", "gpu_w", "ane_w"])
            for s in self.samples:
                w.writerow([f"{s.t_wall:.3f}", f"{s.elapsed_s:.4f}",
                            f"{s.cpu_w:.3f}", f"{s.gpu_w:.3f}", f"{s.ane_w:.3f}"])


class PowerMeter:
    def __init__(self):
        self._proc = None
        self._thread = None
        self._samples = []
        self._t0 = None

    def start(self):
        self._proc = subprocess.Popen(
            CMD, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        # Fail fast if sudo refused (password needed / rule missing).
        time.sleep(0.3)
        if self._proc.poll() is not None:
            err = self._proc.stderr.read()
            raise RuntimeError(f"powermetrics did not start: {err.strip()}")
        return self

    def _read(self):
        cur = None
        for line in self._proc.stdout:
            m = _HEADER.search(line)
            if m:
                cur = PowerSample(t_wall=time.time(), elapsed_s=float(m.group(1)) / 1000)
                continue
            m = _POWER.match(line)
            if m and cur is not None:
                setattr(cur, f"{m.group(1).lower()}_w", int(m.group(2)) / 1000)
                if m.group(1) == "ANE":  # last power line of a sample
                    self._samples.append(cur)
                    cur = None

    def stop(self) -> PowerResult:
        duration = time.time() - self._t0
        self._proc.terminate()  # sudo relays SIGTERM to powermetrics
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._thread.join(timeout=5)
        return PowerResult(duration_s=duration, samples=list(self._samples))
