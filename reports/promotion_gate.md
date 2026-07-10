# Model promotion gate

## Release conclusion: **PASS**

Recommendation: `eligible_for_controlled_release`  
Model version: `0.5.0`  
Selected model: `random_forest`  
Frozen policy threshold: `0.50`

## Evidence checks

| Check | Result |
|---|---|
| auc | PASS |
| brier | PASS |
| expected_calibration_error | PASS |
| precision_at_high_confidence | PASS |
| precision_at_policy_threshold | PASS |
| employment_support_rate_gap | PASS |
| age_support_rate_gap | PASS |
| threshold_selected_on_validation | PASS |
| final_metrics_from_test | PASS |

## Evaluation design

Model selection: `validation`  
Threshold selection: `validation`  
Final evaluation: `test`
