# Model explainability report

This report uses held-out permutation importance to show which inputs most affect predictive performance. It is a global model diagnostic; it does not prove that a feature caused an individual outcome.

| rank | feature | mean importance | standard deviation |
|---:|---|---:|---:|
| 1 | cash_ratio | 0.0998 | 0.0144 |
| 2 | age | 0.0554 | 0.0097 |
| 3 | risk_score | 0.0411 | 0.0073 |
| 4 | accessible_total | 0.0218 | 0.0060 |
| 5 | debt_to_income | 0.0141 | 0.0037 |
| 6 | annual_income | 0.0112 | 0.0044 |
| 7 | cash_balance | 0.0084 | 0.0037 |
| 8 | recent_activity_count | 0.0067 | 0.0029 |
| 9 | investment_balance | 0.0049 | 0.0031 |
| 10 | account_type | 0.0032 | 0.0025 |

The API also separates two explanation types:

- `reason_codes`: model-relevant characteristics derived from the request;
- `policy_reasons`: deterministic rules that produced the final action.

This separation avoids presenting a business rule as though it were a learned model explanation.
