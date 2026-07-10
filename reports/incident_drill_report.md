# Incident drill report

Created at: `2026-07-10T05:33:51.836258+00:00`
Status: `PASS`
Scenarios checked: `4`

## latency_slo_breach

Trigger: p95 latency exceeds 250 ms for 15 minutes
Evidence in demo: p95 latency = 106.14992000000711 ms
First action: freeze release and scale model service replicas
Rollback condition: p95 remains above target after scaling or error rate rises

## prediction_drift

Trigger: drift report status is not ok
Evidence in demo: drift status = ok
First action: route new model to shadow mode and request model-risk review
Rollback condition: high-impact segment drift is confirmed

## promotion_gate_review

Trigger: promotion gate returns REVIEW
Evidence in demo: promotion status = PASS
First action: block production promotion and open approval ticket
Rollback condition: current production model is already exposed to the failing change

## privacy_guard_failure

Trigger: privacy report contains blocked direct identifiers
Evidence in demo: privacy status = PASS
First action: stop data export and remove blocked fields
Rollback condition: blocked fields reached a served endpoint or report artifact
