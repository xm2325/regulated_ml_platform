from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def evaluate_accelerator_policy(
    contract: dict[str, Any],
    policy: dict[str, Any],
    gpu_benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    family = str(contract.get("model_family", "unknown"))
    candidates = {str(value) for value in policy.get("gpu_candidate_families", [])}
    reasons: list[str] = []

    if family not in candidates:
        reasons.append(f"model family '{family}' is not approved as a default GPU candidate")
        return {
            "decision": "CPU_ONLY",
            "model_family": family,
            "gpu_profile_enabled": False,
            "reasons": reasons,
            "evidence": gpu_benchmark,
        }

    if gpu_benchmark is None:
        reasons.append("real GPU benchmark evidence is required before enabling a GPU profile")
        return {
            "decision": "GPU_BENCHMARK_REQUIRED",
            "model_family": family,
            "gpu_profile_enabled": False,
            "reasons": reasons,
            "evidence": None,
        }

    if policy.get("require_real_gpu_runtime_evidence", True) and gpu_benchmark.get("runtime_evidence") != "real_gpu":
        reasons.append("benchmark is not marked as real_gpu runtime evidence")
    if policy.get("require_probability_parity_pass", True) and gpu_benchmark.get("probability_parity_status") != "PASS":
        reasons.append("probability parity did not pass")
    if policy.get("require_policy_decision_parity_pass", True) and gpu_benchmark.get("policy_decision_parity_status") != "PASS":
        reasons.append("policy decision parity did not pass")

    speedup = float(gpu_benchmark.get("throughput_speedup_ratio", 0.0))
    latency_ratio = float(gpu_benchmark.get("gpu_p95_to_cpu_p95_ratio", float("inf")))
    utilization = float(gpu_benchmark.get("sustained_gpu_utilization", 0.0))
    if speedup < float(policy.get("minimum_gpu_speedup_ratio", 1.5)):
        reasons.append("GPU throughput speedup is below the configured minimum")
    if latency_ratio > float(policy.get("maximum_gpu_p95_latency_ratio", 1.1)):
        reasons.append("GPU p95 latency ratio is above the configured maximum")
    if utilization < float(policy.get("minimum_sustained_gpu_utilization", 0.35)):
        reasons.append("GPU utilization is too low to justify dedicated accelerator capacity")
    if utilization > float(policy.get("maximum_sustained_gpu_utilization", 0.90)):
        reasons.append("GPU utilization is too high for safe sustained headroom")

    decision = "GPU_ELIGIBLE" if not reasons else "GPU_REJECTED"
    return {
        "decision": decision,
        "model_family": family,
        "gpu_profile_enabled": decision == "GPU_ELIGIBLE",
        "reasons": reasons,
        "evidence": gpu_benchmark,
        "thresholds": {
            "minimum_gpu_speedup_ratio": float(policy.get("minimum_gpu_speedup_ratio", 1.5)),
            "maximum_gpu_p95_latency_ratio": float(policy.get("maximum_gpu_p95_latency_ratio", 1.1)),
            "minimum_sustained_gpu_utilization": float(policy.get("minimum_sustained_gpu_utilization", 0.35)),
            "maximum_sustained_gpu_utilization": float(policy.get("maximum_sustained_gpu_utilization", 0.90)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="models/triton/contract.json")
    parser.add_argument("--policy", default="config/accelerator_policy.yaml")
    parser.add_argument("--gpu-benchmark")
    parser.add_argument("--output", default="reports/accelerator_decision.json")
    args = parser.parse_args()
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    policy = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8")) or {}
    benchmark = json.loads(Path(args.gpu_benchmark).read_text(encoding="utf-8")) if args.gpu_benchmark else None
    report = evaluate_accelerator_policy(contract, policy, benchmark)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
