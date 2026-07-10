from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_evidence(root: Path) -> dict[str, Any]:
    metrics = _load(root / "reports/model_metrics.json")
    gate = _load(root / "reports/promotion_gate.json")
    load = _load(root / "reports/load_test_summary.json")
    deployment = _load(root / "reports/deployment_validation.json")
    privacy = _load(root / "reports/privacy_report.json")
    quality = _load(root / "reports/data_quality_report.json")
    drift = _load(root / "reports/drift_summary.json")
    best = metrics.get("best_model", "unknown")
    result = metrics.get("models", {}).get(best, {})
    return {
        "release_status": gate.get("status", "UNKNOWN"),
        "model_version": metrics.get("model_version", "unknown"),
        "best_model": best,
        "metrics": {
            "auc": result.get("auc"),
            "brier": result.get("brier"),
            "ece": result.get("expected_calibration_error"),
            "policy_precision": result.get("precision_at_policy_threshold"),
            "policy_recall": result.get("recall_at_policy_threshold"),
            "high_confidence_precision": result.get("precision_at_high_confidence"),
            "policy_threshold": metrics.get("policy_threshold"),
            "p95_latency_ms": load.get("latency_ms_p95"),
        },
        "controls": {
            "promotion": gate.get("status"),
            "data_quality": quality.get("status"),
            "privacy": privacy.get("status"),
            "drift": drift.get("overall_status"),
            "deployment": deployment.get("status"),
            "load_slo": load.get("slo_pass"),
        },
        "split_ranges": metrics.get("split_ranges", {}),
        "calibration_comparison": metrics.get("calibration_comparison", {}),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render(evidence: dict[str, Any]) -> str:
    metrics = evidence["metrics"]
    controls = evidence["controls"]
    rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{html.escape(str(value))}</td></tr>" for name, value in controls.items())
    splits = "".join(f"<tr><td>{html.escape(name)}</td><td>{value.get('n')}</td><td>{value.get('start_date')}</td><td>{value.get('end_date')}</td></tr>" for name, value in evidence["split_ranges"].items())
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Regulated AI MLOps Platform</title><style>body{{font-family:system-ui;margin:0;background:#f5f7fb;color:#172033}}main{{max-width:1100px;margin:auto;padding:32px 20px}}section{{background:white;border:1px solid #dce3ef;border-radius:16px;padding:24px;margin:18px 0}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}.metric{{border:1px solid #dce3ef;border-radius:12px;padding:14px}}.metric b{{display:block;font-size:1.5rem}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #dce3ef;text-align:left}}.pass{{color:#146c43;font-weight:700}}code,pre{{background:#eef1f7;border-radius:8px}}pre{{padding:16px;overflow:auto}}</style></head><body><main><section><p>Version {evidence['model_version']}</p><h1>Controlled ML release evidence</h1><p class='pass'>Release status: {evidence['release_status']}</p><p>The result is shown first, followed by evaluation design, control evidence, and reproduction steps.</p></section><section><h2>Out-of-time result</h2><div class='metrics'><div class='metric'>AUC<b>{_fmt(metrics['auc'])}</b></div><div class='metric'>Brier<b>{_fmt(metrics['brier'])}</b></div><div class='metric'>ECE<b>{_fmt(metrics['ece'])}</b></div><div class='metric'>Policy precision<b>{_fmt(metrics['policy_precision'])}</b></div><div class='metric'>Threshold<b>{_fmt(metrics['policy_threshold'],2)}</b></div><div class='metric'>p95 latency<b>{_fmt(metrics['p95_latency_ms'],1)} ms</b></div></div></section><section><h2>Five chronological windows</h2><table><tr><th>Window</th><th>Rows</th><th>Start</th><th>End</th></tr>{splits}</table></section><section><h2>Control evidence</h2><table><tr><th>Control</th><th>Status</th></tr>{rows}</table></section><section><h2>Decision path</h2><pre>request → features → calibrated probability → versioned policy → review route → audit IDs</pre></section><section><h2>Reproduce</h2><pre>pip install -r requirements-dev.txt\nmake evidence\nmake lint\nmake security\nmake audit</pre><p>The dataset and actions are synthetic and must not be used for real financial decisions.</p></section></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="site")
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    evidence = build_evidence(root)
    (output / "data/evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (output / "index.html").write_text(render(evidence), encoding="utf-8")
    print(f"Wrote evidence dashboard to {output}")


if __name__ == "__main__":
    main()
