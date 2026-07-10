# Model card: targeted-support demonstration model

## Release status

**Automated promotion gate: PASS for a controlled synthetic-data demonstration.**

This status does not approve use for real customer decisions.

## Intended use

The model estimates a synthetic probability that a customer record belongs to a generated `support_needed` class. The project tests production ML engineering: training, versioning, serving, review routing, monitoring, deployment, and release evidence.

## Not intended for

- financial advice;
- credit approval or decline;
- customer vulnerability classification;
- real-customer decisions;
- use with personal data without privacy, security, legal, and model-risk review.

## Model and policy separation

The random-forest model returns a probability. A separate deterministic policy maps the score and hard safety gates to one of four actions:

- `no_support`
- `cash_buffer_warning`
- `investment_support`
- `risk_review`

Borderline and safety-sensitive records can be routed to `manual_review`. The model does not directly issue the final action.

## Versions

| Part | Version |
|---|---|
| model | `0.5.0` |
| policy | `targeted-support-policy-v2` |
| feature schema | `financial_customer_features_v3` |
| contract | `model-contract-v1` |

## Evaluation design

Customer IDs are assigned deterministically to train, validation, and test splits. Preprocessing and model fitting use training data. Champion selection and threshold selection use validation data. The test split is used only after both are frozen.

## Independent test result

| Metric | Value |
|---|---:|
| AUC | 0.7847 |
| average precision | 0.8385 |
| Brier score | 0.1887 |
| expected calibration error | 0.0767 |
| policy precision | 0.8220 |
| policy recall | 0.6716 |
| high-confidence precision | 0.8792 |

Bootstrap 95% intervals:

- AUC: 0.756–0.817
- Brier score: 0.177–0.200
- policy precision: 0.790–0.859

## Main risks

- synthetic labels may not represent a real customer need;
- reason codes are diagnostic descriptions, not causal findings;
- segment metrics can hide within-group failure modes;
- fixed thresholds may degrade after distribution change;
- manual-review rules require accountable operational owners;
- a model score must not be described as advice or a guarantee.

## Required monitoring

- request count, latency, error rate, and SLO budget;
- feature missingness and range violations;
- prediction and feature drift;
- action and manual-review rates;
- calibration and performance when delayed labels become available;
- segment-level precision, recall, support rate, and error analysis;
- model, policy, and schema version distribution.

## Rollback triggers

Rollback or freeze promotion when a required release control fails, p95 latency remains above target, drift is linked to performance loss, a privacy control fails, or the policy produces unsafe or unexplained action changes.
