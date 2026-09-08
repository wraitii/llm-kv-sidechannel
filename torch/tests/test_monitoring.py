import json
from pathlib import Path

from llmpr_torch.monitoring import SystemMonitor, system_sample


def test_system_sample_survives_missing_nvidia_smi(tmp_path: Path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr("llmpr_torch.monitoring.subprocess.run", missing)
    sample = system_sample(tmp_path)
    assert sample["event"] == "system"
    assert sample["disk_free_bytes"] > 0
    assert "nvidia_smi_error" in sample


def test_system_monitor_writes_jsonl(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "llmpr_torch.monitoring.system_sample",
        lambda run_dir: {"event": "system", "disk": str(run_dir)},
    )
    monitor = SystemMonitor(tmp_path, 0.01)
    monitor.start()
    monitor.stop_event.wait(0.03)
    monitor.close()
    rows = [json.loads(line) for line in monitor.output.read_text().splitlines()]
    assert rows
    assert all(row["event"] == "system" for row in rows)
