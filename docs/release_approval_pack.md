# Release approval pack

## Release decision: **PASS**

Recommendation: `eligible_for_controlled_demo_release`  
Model: `random_forest / 0.5.0`  
Policy: `targeted-support-policy-v2`  
Feature schema: `financial_customer_features_v3`

## Control status

| Control | Status |
|---|---|
| promotion | PASS |
| data_quality | PASS |
| privacy | PASS |
| drift | OK |
| load_test | PASS |
| incident_drill | PASS |
| deployment | PASS |
| reproducibility | PASS |

## Key evidence

| Metric | Value |
|---|---:|
| auc | 0.7847 |
| brier | 0.1887 |
| expected_calibration_error | 0.0767 |
| precision_at_policy_threshold | 0.8220 |
| precision_at_high_confidence | 0.8792 |
| load_test_p95_ms | 106.1499 |

## Evaluation design

- model_selection: `validation`
- threshold_selection: `validation`
- final_evaluation: `test`

## Evidence files

- `reports/model_evaluation.md`
- `reports/calibration_report.md`
- `reports/fairness_report.md`
- `reports/drift_report.html`
- `reports/data_quality_report.json`
- `reports/privacy_report.md`
- `reports/promotion_gate.md`
- `reports/champion_challenger_report.md`
- `reports/explainability_report.md`
- `reports/load_test_report.md`
- `reports/incident_drill_report.md`
- `reports/deployment_validation.md`
- `docs/model_contract.md`
- `docs/reproducibility_manifest.md`
- `docs/rollback_policy.md`
- `docs/slo_error_budget.md`

## Boundary

This is a synthetic engineering release pack. Real customer use would require independent validation, security testing, data-protection review, model-risk approval, legal review, and accountable business sign-off.
