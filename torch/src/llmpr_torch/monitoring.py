"""Low-overhead host and accelerator sampling for training runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import psutil


def system_sample(run_dir: Path) -> dict:
    memory = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    disk = shutil.disk_usage(run_dir)
    sample = {
        "event": "system", "time": time.time(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_used_bytes": memory.used,
        "memory_available_bytes": memory.available,
        "process_rss_bytes": process.memory_info().rss,
        "disk_used_bytes": disk.used, "disk_free_bytes": disk.free,
    }
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,power.limit", "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True)
        values = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
        sample.update(dict(zip((
            "gpu_utilization_percent", "gpu_memory_used_mib", "gpu_memory_total_mib",
            "gpu_temperature_c", "gpu_power_w", "gpu_power_limit_w",
        ), values, strict=True)))
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as error:
        sample["nvidia_smi_error"] = str(error)
    return sample


class SystemMonitor(threading.Thread):
    def __init__(self, run_dir: Path, interval_s: float):
        super().__init__(name="llmpr-system-monitor", daemon=True)
        self.run_dir = run_dir
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.output = run_dir / "system.jsonl"

    def run(self) -> None:
        while not self.stop_event.is_set():
            sample = system_sample(self.run_dir)
            with self.output.open("a") as handle:
                handle.write(json.dumps(sample) + "\n")
            self.stop_event.wait(self.interval_s)

    def close(self) -> None:
        self.stop_event.set()
        self.join(timeout=max(1.0, self.interval_s + 1.0))
