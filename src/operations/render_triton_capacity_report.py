from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _chart(
    title: str,
    series: list[tuple[str, list[tuple[float, float]]]],
    y_label: str,
    thresholds: list[tuple[str, float]] | None = None,
) -> str:
    width, height, padding = 760, 320, 52
    thresholds = thresholds or []
    all_points = [point for _, values in series for point in values]
    if not all_points:
        return ""

    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points] + [value for _, value in thresholds]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(0.0, min(ys)), max(ys)
    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0
    max_y *= 1.08

    def scale_x(value: float) -> float:
        return padding + (value - min_x) / (max_x - min_x) * (width - 2 * padding)

    def scale_y(value: float) -> float:
        return height - padding - (value - min_y) / (max_y - min_y) * (height - 2 * padding)

    paths: list[str] = []
    legend: list[str] = []
    dash_patterns = ["", "7 5", "2 4"]
    for index, (name, values) in enumerate(series):
        if not values:
            continue
        points = " ".join(f"{scale_x(x):.1f},{scale_y(y):.1f}" for x, y in values)
        dash = dash_patterns[index % len(dash_patterns)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        paths.append(
            f'<polyline points="{points}" fill="none" stroke="currentColor" stroke-width="2.5"{dash_attr}/>'
        )
        for x, y in values:
            paths.append(f'<circle cx="{scale_x(x):.1f}" cy="{scale_y(y):.1f}" r="3.2" fill="currentColor"/>')
        legend.append(f"<span>{html.escape(name)}</span>")

    threshold_lines: list[str] = []
    for name, value in thresholds:
        y = scale_y(value)
        threshold_lines.append(
            f'<line x1="{padding}" y1="{y:.1f}" x2="{width-padding}" y2="{y:.1f}" '
            'stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 5" opacity="0.55"/>'
            f'<text x="{width-padding-4}" y="{y-5:.1f}" text-anchor="end" font-size="11">{html.escape(name)}</text>'
        )

    unique_xs = sorted(set(xs))
    x_ticks = "".join(
        f'<text x="{scale_x(value):.1f}" y="{height-padding+20}" text-anchor="middle" font-size="11">{value:g}</text>'
        for value in unique_xs
    )
    y_ticks = "".join(
        f'<text x="{padding-8}" y="{scale_y(value)+4:.1f}" text-anchor="end" font-size="11">{value:.1f}</text>'
        for value in [min_y, (min_y + max_y) / 2, max_y]
    )

    return f"""
<section class="chart-card">
  <h3>{html.escape(title)}</h3>
  <p class="axis-label">{html.escape(y_label)} vs concurrency · shared axis across all series</p>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
    <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="currentColor" opacity="0.25"/>
    <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="currentColor" opacity="0.25"/>
    {''.join(threshold_lines)}
    {''.join(paths)}
    {x_ticks}
    {y_ticks}
  </svg>
  <div class="legend">{' · '.join(legend)}</div>
</section>
"""


def render_report(benchmark: dict[str, Any], capacity: dict[str, Any]) -> str:
    scenarios = benchmark.get("scenarios", [])
    perf_points = capacity.get("perf_analyzer", {}).get("points", [])
    semantic_latency = [(float(item["concurrency"]), float(item["p95_latency_ms"])) for item in scenarios]
    semantic_throughput = [(float(item["concurrency"]), float(item["rows_per_second"])) for item in scenarios]
    average_batch = [
        (float(item["concurrency"]), float(item.get("support_base_average_batch_size", 0.0))) for item in scenarios
    ]
    perf_latency = [
        (float(item["concurrency"]), float(item["p95_latency_ms"]))
        for item in perf_points
        if item.get("concurrency") is not None and item.get("p95_latency_ms") is not None
    ]
    perf_throughput = [
        (float(item["concurrency"]), float(item["inferences_per_second"]))
        for item in perf_points
        if item.get("concurrency") is not None and item.get("inferences_per_second") is not None
    ]

    cap = capacity.get("capacity_evidence", {})
    batching = capacity.get("batching_evidence", {})
    objectives = capacity.get("service_objectives", {})
    rows = "".join(
        "<tr>"
        f"<td>{int(item['concurrency'])}</td>"
        f"<td>{float(item['p95_latency_ms']):.2f}</td>"
        f"<td>{float(item['p99_latency_ms']):.2f}</td>"
        f"<td>{float(item['rows_per_second']):.1f}</td>"
        f"<td>{float(item.get('support_base_average_batch_size', 0.0)):.2f}</td>"
        f"<td>{html.escape(item.get('parity_status', ''))}</td>"
        "</tr>"
        for item in scenarios
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Triton v1.2 Capacity Evidence</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f6f7fb;color:#172033}}main{{max-width:1120px;margin:auto;padding:30px 18px}}
section{{background:white;border:1px solid #d9dfeb;border-radius:14px;padding:22px;margin:16px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.metric{{border:1px solid #d9dfeb;border-radius:10px;padding:14px}}.metric b{{display:block;font-size:1.6rem;margin-top:6px}}.charts{{display:grid;gap:16px}}
.chart-card svg{{width:100%;height:auto;background:#fafbfe;border-radius:10px;color:#243b66}}.legend{{font-size:.9rem;color:#586277}}.axis-label{{color:#586277}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e3e7ef;text-align:right}}th:first-child,td:first-child{{text-align:left}}
code,pre{{background:#eef1f7;border-radius:6px}}pre{{padding:14px;overflow:auto}}
</style></head><body><main>
<section><h1>Triton v1.2 concurrency and capacity evidence</h1><p><b>Status: {html.escape(capacity.get('status','UNKNOWN'))}</b> · Decision: <code>{html.escape(capacity.get('decision',''))}</code></p><p>{html.escape(capacity.get('reason',''))}</p></section>
<section><h2>Result first</h2><div class="grid">
<div class="metric">Observed max avg batch<b>{float(batching.get('maximum_observed_average_batch_size',0.0)):.2f}</b></div>
<div class="metric">Perf Analyzer best concurrency<b>{cap.get('best_slo_passing_concurrency')}</b></div>
<div class="metric">Measured server infer/s<b>{float(cap.get('measured_inferences_per_second',0.0)):.0f}</b></div>
<div class="metric">Safe ref infer/s per replica<b>{float(cap.get('safe_reference_rows_per_second_per_replica',0.0)):.0f}</b></div>
<div class="metric">Reference replicas<b>{cap.get('recommended_reference_replicas')}</b></div>
</div></section>
<section><h2>Evidence roles</h2><pre>custom concurrent HTTP → probability parity + HTTP correctness + observed batching
NVIDIA Triton Perf Analyzer → server latency/throughput capacity source
capacity policy → SLO filter + safety headroom + bounded replica reference</pre></section>
<div class="charts">
{_chart('p95 latency', [('Custom semantic client', semantic_latency), ('Perf Analyzer', perf_latency)], 'milliseconds', [('p95 SLO', float(objectives.get('p95_latency_ms', 20.0)))])}
{_chart('Throughput', [('Custom semantic client', semantic_throughput), ('Perf Analyzer', perf_throughput)], 'rows / inferences per second')}
{_chart('Observed dynamic batch size', [('support_base average batch', average_batch)], 'average rows per backend execution', [('minimum batching gain', float(batching.get('minimum_required_gain', 1.25)))])}
</div>
<section><h2>Custom concurrent HTTP evidence</h2><table><thead><tr><th>Concurrency</th><th>p95 ms</th><th>p99 ms</th><th>Rows/s</th><th>Avg batch</th><th>Parity</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Boundary</h2><p>{html.escape(capacity.get('claim_boundary',{}).get('statement',''))}</p><p>The current validated tree ensemble remains CPU_ONLY. This report is not a production capacity or GPU performance claim.</p></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--capacity", required=True)
    parser.add_argument("--output", default="reports/triton_capacity/capacity_report.html")
    args = parser.parse_args()
    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    capacity = json.loads(Path(args.capacity).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(benchmark, capacity), encoding="utf-8")
    print(f"Wrote Triton capacity report to {output}")


if __name__ == "__main__":
    main()
