# Model evaluation report

**Release conclusion:** the model passed the automated promotion criteria on the independent test split.

Training timestamp: `2026-07-10T05:32:33.119783+00:00`  
Model version: `0.5.0`  
Selected model: `random_forest`  
Frozen policy threshold: `0.50`

## Independent test result

| Metric | Value |
|---|---:|
| auc | 0.7847 |
| average_precision | 0.8385 |
| brier | 0.1887 |
| expected_calibration_error | 0.0767 |
| precision_at_policy_threshold | 0.8220 |
| recall_at_policy_threshold | 0.6716 |
| policy_support_rate | 0.5000 |
| precision_at_0_5 | 0.8220 |
| high_confidence_rate | 0.3500 |
| positive_high_confidence_rate | 0.2650 |
| precision_at_high_confidence | 0.8792 |

## Bootstrap uncertainty

| Metric | 95% lower | Median | 95% upper |
|---|---:|---:|---:|
| auc | 0.7562 | 0.7858 | 0.8168 |
| brier | 0.1767 | 0.1884 | 0.2004 |
| precision_at_policy_threshold | 0.7905 | 0.8223 | 0.8586 |

## Leakage control

Model selection and threshold choice use the validation split. The test split is held back until the model and threshold are fixed.

## Validation threshold search

| threshold | precision | recall | support_rate | f1 |
|---|---|---|---|---|
| 0.3000 | 0.6722 | 0.9183 | 0.8360 | 0.7762 |
| 0.3500 | 0.7194 | 0.8464 | 0.7200 | 0.7778 |
| 0.4000 | 0.7635 | 0.7859 | 0.6300 | 0.7746 |
| 0.4500 | 0.7887 | 0.7320 | 0.5680 | 0.7593 |
| 0.5000 | 0.8157 | 0.6797 | 0.5100 | 0.7415 |
| 0.5500 | 0.8330 | 0.6601 | 0.4850 | 0.7366 |
| 0.6000 | 0.8370 | 0.6291 | 0.4600 | 0.7183 |
| 0.6500 | 0.8421 | 0.5752 | 0.4180 | 0.6835 |
| 0.7000 | 0.8642 | 0.4886 | 0.3460 | 0.6242 |
| 0.7500 | 0.8765 | 0.3595 | 0.2510 | 0.5098 |
| 0.8000 | 0.8784 | 0.2124 | 0.1480 | 0.3421 |
| 0.8500 | 0.8889 | 0.0915 | 0.0630 | 0.1659 |
| 0.9000 | 0.9444 | 0.0278 | 0.0180 | 0.0540 |

## Boundary

The dataset and outcome are synthetic. The report tests the engineering and governance path; it is not evidence that the policy is suitable for real financial advice.
