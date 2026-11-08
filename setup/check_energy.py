"""Check that GPU energy is measured on this Mac.

Runs ~30 s of ResNet-18 training steps on MPS inside `measure_run` and prints
the power meter's and CodeCarbon's numbers. Expect GPU average power of several
watts from the meter (it was 7-9 W in raw powermetrics on an M2).

Usage: python setup/check_energy.py
"""
import sys
import time
from pathlib import Path

import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.energy import measure_run  # noqa: E402


def main():
    assert torch.backends.mps.is_available(), "MPS not available"
    dev = torch.device("mps")
    torch.manual_seed(0)
    model = torchvision.models.resnet18(num_classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    x = torch.randn(64, 3, 128, 128, device=dev)
    y = torch.randn(64, 1, device=dev)

    with measure_run("setup_check_energy", phase="setup", model="resnet18",
                     save=False) as m:
        t0 = time.time()
        steps = 0
        while time.time() - t0 < 30:
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(x), y).backward()
            opt.step()
            steps += 1
        torch.mps.synchronize()

    r = m.row
    d = r["duration_s"]
    w = lambda kwh: kwh * 3.6e6 / d  # noqa: E731  average watts
    print(f"steps: {steps}  duration s: {d}  meter coverage: {r['meter_coverage']}")
    print(f"meter avg W   cpu {w(r['meter_cpu_kwh']):.2f}  gpu {w(r['meter_gpu_kwh']):.2f}"
          f"  ane {w(r['meter_ane_kwh']):.2f}")
    print(f"meter  total kWh {r['meter_total_kwh']:.6f}  kg CO2eq {r['meter_kg_co2eq']:.7f}")
    print(f"codecarbon kWh   {r['cc_energy_kwh']:.6f}  kg CO2eq {r['cc_kg_co2eq']:.7f}")
    ok = w(r["meter_gpu_kwh"]) > 1.0
    print("GPU energy measured:", "OK" if ok else "NO (GPU average below 1 W)")


if __name__ == "__main__":
    main()
