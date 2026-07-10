# Model and policy rollback policy

## Rollback unit

The release unit is the combination of:

- container image digest;
- model artifact checksum and model version;
- policy version and frozen threshold;
- feature-schema version;
- OpenAPI and decision-contract version;
- release evidence manifest.

Rolling back only the model while leaving an incompatible policy or schema in place is not allowed.

## Immediate rollback triggers

- hard safety-gate logic differs from the approved policy;
- privacy or direct-identifier check fails;
- model, policy, or schema identifier is missing from a served response;
- sustained p95 latency or error-rate breach after normal scaling;
- a release produces an unexplained increase in action changes or manual-review volume;
- confirmed drift is linked to a material performance or calibration loss;
- the production image cannot pass readiness or the decision-contract check.

## Procedure

1. Freeze further promotion and preserve the failing release identifiers.
2. Shift traffic to the last approved blue or stable deployment.
3. Confirm `/ready`, `/version`, and `/decision-contract` on the restored release.
4. Compare decision and audit samples before and after rollback.
5. Record the incident, trigger, affected interval, versions, owner, and corrective action.
6. Re-run data quality, model evaluation, policy tests, deployment validation, load test, and release gate before a new promotion.

## Evidence required before closure

- incident timeline;
- last approved and failing image digests;
- model, policy, and feature-schema versions;
- affected request and audit identifiers;
- metrics before, during, and after rollback;
- root-cause statement;
- test and approval evidence for the corrective release.
