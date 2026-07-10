# Segment behaviour report

Frozen policy threshold: `0.50`

This is a model-risk diagnostic, not a legal fairness conclusion. It surfaces where segment-level review is needed.

## age_band

| group | n | mean_probability | observed_positive_rate | predicted_support_rate | precision_at_policy_threshold | recall_at_policy_threshold | auc | brier |
|---|---|---|---|---|---|---|---|---|
| 18-30 | 237 | 0.5183 | 0.5148 | 0.4979 | 0.7458 | 0.7213 | 0.7668 | 0.1941 |
| 31-45 | 251 | 0.5360 | 0.6016 | 0.4940 | 0.7903 | 0.6490 | 0.7439 | 0.2025 |
| 46-60 | 252 | 0.5653 | 0.7143 | 0.5317 | 0.9328 | 0.6944 | 0.8513 | 0.1643 |
| 61+ | 260 | 0.5337 | 0.6115 | 0.4769 | 0.8065 | 0.6289 | 0.7831 | 0.1940 |

Largest between-group gaps: predicted_support_rate_gap=0.0548, mean_probability_gap=0.0470, precision_gap=0.1871, auc_gap=0.1074

## income_band

| group | n | mean_probability | observed_positive_rate | predicted_support_rate | precision_at_policy_threshold | recall_at_policy_threshold | auc | brier |
|---|---|---|---|---|---|---|---|---|
| Q1 | 250 | 0.5421 | 0.6120 | 0.5160 | 0.7907 | 0.6667 | 0.7803 | 0.1929 |
| Q2 | 250 | 0.5117 | 0.5720 | 0.4360 | 0.8440 | 0.6434 | 0.8224 | 0.1802 |
| Q3 | 250 | 0.5534 | 0.6520 | 0.5200 | 0.8462 | 0.6748 | 0.7562 | 0.1953 |
| Q4 | 250 | 0.5471 | 0.6120 | 0.5280 | 0.8106 | 0.6993 | 0.7819 | 0.1863 |

Largest between-group gaps: predicted_support_rate_gap=0.0920, mean_probability_gap=0.0418, precision_gap=0.0555, auc_gap=0.0662

## employment_status

| group | n | mean_probability | observed_positive_rate | predicted_support_rate | precision_at_policy_threshold | recall_at_policy_threshold | auc | brier |
|---|---|---|---|---|---|---|---|---|
| employed | 591 | 0.5461 | 0.6108 | 0.5212 | 0.8182 | 0.6981 | 0.7943 | 0.1835 |
| retired | 163 | 0.5253 | 0.6196 | 0.4356 | 0.8732 | 0.6139 | 0.7847 | 0.1903 |
| self_employed | 102 | 0.5232 | 0.6373 | 0.4902 | 0.7800 | 0.6000 | 0.7214 | 0.2183 |
| student | 66 | 0.5218 | 0.5909 | 0.4848 | 0.8750 | 0.7179 | 0.8557 | 0.1665 |
| unemployed | 78 | 0.5443 | 0.5897 | 0.5000 | 0.7692 | 0.6522 | 0.7466 | 0.2049 |

Largest between-group gaps: predicted_support_rate_gap=0.0856, mean_probability_gap=0.0243, precision_gap=0.1058, auc_gap=0.1342
