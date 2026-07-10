# Model and decision contract

## Contract conclusion

The API contract fixes the model, feature schema, policy, allowed actions, review route, and audit fields as separate versioned parts.

- Contract version: `model-contract-v1`
- Model version: `0.5.0`
- Policy version: `targeted-support-policy-v2`
- Feature schema version: `financial_customer_features_v3`
- Frozen threshold: `0.5`

## Input fields

| Field | Type | Required |
|---|---|---|
| `customer_id` | `string` | yes |
| `request_id` | `[{'maxLength': 128, 'minLength': 8, 'type': 'string'}, {'type': 'null'}]` | no |
| `age` | `integer` | yes |
| `annual_income` | `number` | yes |
| `cash_balance` | `number` | yes |
| `investment_balance` | `number` | yes |
| `debt_balance` | `number` | yes |
| `risk_score` | `number` | yes |
| `recent_activity_count` | `integer` | yes |
| `account_type` | `string` | yes |
| `employment_status` | `string` | yes |

## Prohibited uses

- financial advice
- credit approval or decline
- customer vulnerability classification
- use with real personal data without a data-protection and model-risk review
