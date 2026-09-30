"""Cost and feasibility decisions at Breeze's movable CPU/GPU boundary."""

import pytest

from rho_tts.providers.breeze_schedule import (
    DeviceCost,
    choose_device,
    reference_resample_costs,
)


def test_scheduler_includes_queue_and_both_transfers():
    costs = {
        "cpu": DeviceCost(compute_ms=4.0),
        "cuda": DeviceCost(
            compute_ms=0.5, transfer_in_ms=0.3, transfer_out_ms=0.2,
            queue_ms=4.0, required_bytes=100, available_bytes=100,
        ),
    }
    assert choose_device(costs) == "cpu"
    costs["cuda"] = DeviceCost(
        compute_ms=0.5, transfer_in_ms=0.3, transfer_out_ms=0.2,
        required_bytes=100, available_bytes=100,
    )
    assert choose_device(costs) == "cuda"


def test_scheduler_rejects_unsupported_and_insufficient_memory():
    costs = {
        "cpu": DeviceCost(compute_ms=4.0),
        "cuda": DeviceCost(compute_ms=0.1, required_bytes=101, available_bytes=100),
    }
    assert choose_device(costs) == "cpu"
    costs["cuda"] = DeviceCost(compute_ms=0.1, supported=False)
    assert choose_device(costs) == "cpu"
    with pytest.raises(RuntimeError, match="No feasible device"):
        choose_device({"cuda": costs["cuda"]})


def test_measured_reference_routes_to_gpu_when_uncontended():
    costs = reference_resample_costs(
        source_rate=48000, target_rate=24000, samples=634_424,
        gpu_name="NVIDIA GeForce RTX 3060", gpu_free_bytes=2 * 1024**3,
    )
    assert choose_device(costs) == "cuda"
    assert costs["cuda"].predicted_ms < costs["cpu"].predicted_ms
    assert choose_device(reference_resample_costs(
        source_rate=48000, target_rate=24000, samples=634_424,
        gpu_name="NVIDIA GeForce RTX 3060", gpu_free_bytes=2 * 1024**3,
        gpu_queue_ms=10.0,
    )) == "cpu"


@pytest.mark.parametrize("rate,samples,gpu_name,free_bytes", [
    (44100, 634_424, "NVIDIA GeForce RTX 3060", 2 * 1024**3),
    (48000, 10_000, "NVIDIA GeForce RTX 3060", 2 * 1024**3),
    (48000, 634_424, "Other GPU", 2 * 1024**3),
    (48000, 634_424, "NVIDIA GeForce RTX 3060", 0),
])
def test_unmeasured_or_low_memory_reference_stays_on_cpu(rate, samples, gpu_name, free_bytes):
    costs = reference_resample_costs(
        source_rate=rate, target_rate=24000, samples=samples,
        gpu_name=gpu_name, gpu_free_bytes=free_bytes,
    )
    assert choose_device(costs) == "cpu"
