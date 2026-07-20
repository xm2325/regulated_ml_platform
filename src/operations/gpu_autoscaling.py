from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def decide_gpu_scaling(
    accelerator_decision: dict[str, Any],
    metrics: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    required = str(policy.get("require_accelerator_decision", "GPU_ELIGIBLE"))
    current_replicas = int(metrics.get("replicas", 1))
    min_replicas = int(policy.get("min_replicas", 1))
    max_replicas = int(policy.get("max_replicas", 8))
    gpu_utilization = float(metrics.get("gpu_utilization", 0.0))
    queue_ms = float(metrics.get("queue_time_ms_per_request", 0.0))
    average_batch_size = float(metrics.get("average_batch_size", 0.0))

    if accelerator_decision.get("decision") != required:
        return {
            "decision": "GPU_PROFILE_DISABLED",
            "desired_replicas": current_replicas,
            "reason": f"accelerator decision must be {required}",
            "metrics": metrics,
        }

    scale_out = policy.get("scale_out", {})
    scale_in = policy.get("scale_in", {})
    high_gpu = gpu_utilization >= float(scale_out.get("gpu_utilization_at_or_above", 0.85))
    high_queue = queue_ms >= float(scale_out.get("queue_time_ms_at_or_above", 8.0))
    low_gpu = gpu_utilization <= float(scale_in.get("gpu_utilization_at_or_below", 0.30))
    low_queue = queue_ms <= float(scale_in.get("queue_time_ms_at_or_below", 1.0))
    batching_present = average_batch_size >= float(scale_in.get("minimum_average_batch_size", 2.0))

    if (high_gpu or high_queue) and current_replicas < max_replicas:
        reason = "queue pressure" if high_queue else "sustained GPU utilization"
        return {
            "decision": "SCALE_OUT",
            "desired_replicas": current_replicas + 1,
            "reason": reason,
            "metrics": metrics,
            "cooldown_seconds": int(policy.get("cooldown_seconds", {}).get("scale_out", 120)),
        }

    if low_gpu and low_queue and batching_present and current_replicas > min_replicas:
        return {
            "decision": "SCALE_IN",
            "desired_replicas": current_replicas - 1,
            "reason": "sustained spare GPU capacity with no queue pressure and batching already active",
            "metrics": metrics,
            "cooldown_seconds": int(policy.get("cooldown_seconds", {}).get("scale_in", 600)),
        }

    reasons: list[str] = []
    if current_replicas >= max_replicas and (high_gpu or high_queue):
        reasons.append("at maximum replica boundary")
    if low_gpu and low_queue and not batching_present:
        reasons.append("low utilization may be caused by weak batching; do not scale in before fixing request aggregation")
    if not reasons:
        reasons.append("metrics are within the configured hold region")
    return {
        "decision": "HOLD",
        "desired_replicas": current_replicas,
        "reason": "; ".join(reasons),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator-decision", default="reports/accelerator_decision.json")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--policy", default="config/gpu_autoscaling.yaml")
    parser.add_argument("--output", default="reports/gpu_autoscaling_decision.json")
    args = parser.parse_args()
    accelerator = json.loads(Path(args.accelerator_decision).read_text(encoding="utf-8"))
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    policy = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8")) or {}
    report = decide_gpu_scaling(accelerator, metrics, policy)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
