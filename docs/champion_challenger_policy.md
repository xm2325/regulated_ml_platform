# Champion-challenger and rollback policy

The platform keeps model training, registration, approval, serving, and rollback as separate actions. A newly trained release is never made customer-facing by the training command alone.

## Registry aliases

| Alias | Meaning | May serve customer-facing traffic? |
|---|---|---|
| `challenger` | Most recently registered release awaiting review or a failed champion returned for investigation | No |
| `champion` | Approved release selected for serving | Yes |
| `rollback` | Last approved champion retained as the immediate recovery target | No, unless rollback is invoked |

Aliases point to immutable MLflow model versions. PostgreSQL stores experiment and registry metadata; MinIO stores model and release artifacts.

## Controlled promotion

Promotion is allowed only when `reports/promotion_gate.json` contains both:

```json
{
  "status": "PASS",
  "release_recommendation": "eligible_for_controlled_release"
}
```

The promotion command also supports an expected challenger version. This prevents a stale approval from promoting a different model version after the reviewer completed the assessment.

When promotion succeeds:

1. the current `champion`, when present, becomes `rollback`;
2. the approved `challenger` becomes `champion`;
3. the `challenger` alias is removed;
4. version tags record approval status and promotion time.

## Rollback

Rollback requires a non-empty incident or change reason. The `rollback` version becomes `champion`; the failed champion becomes `challenger` for investigation. Version tags record the reason, rollback time, and lifecycle state.

The rollback command does not retrain, recalibrate, or change artifacts. It changes only registry aliases and audit tags, so recovery uses an already registered immutable release.

## Evidence used for approval

The automated report compares out-of-time AUC, Brier score, expected calibration error, policy precision, coverage, group behaviour, drift, explainability output, load-test results, deployment controls, dependency audit, and the release manifest. A production approval would also require named model-owner and model-risk sign-off, incident review, and change-management approval.
