"""Cost-based device selection for independently movable Breeze operations.

Costs include the whole CPU -> device -> CPU boundary. A caller must only
advertise a device when the operation has a real implementation there and its
inputs/outputs can cross that boundary safely.
"""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DeviceCost:
    compute_ms: float
    queue_ms: float = 0.0
    transfer_in_ms: float = 0.0
    transfer_out_ms: float = 0.0
    required_bytes: int = 0
    available_bytes: int = 0
    supported: bool = True

    @property
    def predicted_ms(self) -> float:
        return self.queue_ms + self.transfer_in_ms + self.compute_ms + self.transfer_out_ms

    @property
    def feasible(self) -> bool:
        return self.supported and self.required_bytes <= self.available_bytes


def choose_device(costs: Mapping[str, DeviceCost]) -> str:
    """Pick the lowest predicted latency among supported, memory-fitting devices.

    Dict insertion order breaks ties, so callers can list their safe baseline
    first. No device is selected when all implementations are infeasible.
    """
    feasible = ((name, cost) for name, cost in costs.items() if cost.feasible)
    try:
        return min(feasible, key=lambda item: item[1].predicted_ms)[0]
    except ValueError as exc:
        raise RuntimeError("No feasible device for Breeze operation") from exc


# RTX 3060 12 GiB, warmed PyTorch 2.9.1, 634,424 mono float32 samples,
# 48 -> 24 kHz, 30 repeats. See scratch/breeze-schedule-results.json.
# These estimates are intentionally limited to this measured shape and device.
REFERENCE_48K_TO_24K_RTX3060 = {
    "cpu_compute_ms": 4.1033,
    "gpu_transfer_in_ms": 0.3094,
    "gpu_compute_ms": 0.5646,
    "gpu_transfer_out_ms": 0.2406,
}


def reference_resample_costs(
    *,
    source_rate: int,
    target_rate: int,
    samples: int,
    gpu_name: str | None,
    gpu_free_bytes: int = 0,
    gpu_queue_ms: float = 0.0,
) -> dict[str, DeviceCost]:
    """Return measured costs for the reference-resample boundary.

    Keep CPU as the only eligible device outside the measured rate, device,
    and input-size band. Linear sample scaling is used only within 0.5-2x
    of the benchmark; queue cost can override the GPU preference.
    """
    measured_samples = 634_424
    scale = samples / measured_samples
    measured = REFERENCE_48K_TO_24K_RTX3060
    costs = {"cpu": DeviceCost(compute_ms=measured["cpu_compute_ms"] * scale)}
    if (
        source_rate == 48000
        and target_rate == 24000
        and gpu_name == "NVIDIA GeForce RTX 3060"
        and 0.5 <= scale <= 2.0
    ):
        # FFT/resampler temporaries are implementation-dependent; reserve
        # 64 MiB beyond input and output to avoid scheduling on a full GPU.
        required = 64 * 1024**2 + 4 * (samples + (samples + 1) // 2)
        costs["cuda"] = DeviceCost(
            compute_ms=measured["gpu_compute_ms"] * scale,
            transfer_in_ms=measured["gpu_transfer_in_ms"] * scale,
            transfer_out_ms=measured["gpu_transfer_out_ms"] * scale,
            queue_ms=gpu_queue_ms,
            required_bytes=required,
            available_bytes=gpu_free_bytes,
        )
    return costs
