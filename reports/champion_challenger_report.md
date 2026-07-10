# Champion-challenger report

Status: `PROMOTE_CHAMPION`
Reason: Champion has the best AUC among available candidates.

| Model | AUC | Brier | ECE | Precision at policy threshold |
|---|---:|---:|---:|---:|
| random_forest | 0.7847 | 0.1887 | 0.0767 | 0.8220 |
| logistic_regression | 0.7397 | 0.2067 | 0.0923 | 0.7844 |

The challenger is scored in shadow mode before any production action changes. The shadow path records probability deltas and action changes but keeps the champion response as the served output.
