# Service level objectives and error budget

## Objective

The demonstration API targets reliable online scoring while preserving a clear stop condition when service quality degrades.

## Service indicators

| Indicator | Target |
|---|---:|
| successful request rate | at least 99.5% |
| p95 response latency | at most 250 ms |
| readiness success | 100% for pods receiving traffic |
| model artifact load failure | 0 tolerated |
| unversioned decision response | 0 tolerated |

## Error budget

For a 30-day window, a 99.5% successful-request objective allows 0.5% failed requests. The budget is measured separately from expected client-side validation errors.

Promotion is frozen when:

- more than 50% of the monthly error budget is consumed in seven days;
- p95 latency exceeds 250 ms for 15 minutes;
- readiness failures affect more than one replica;
- any response omits model, policy, schema, decision, or audit identifiers;
- the model artifact or policy configuration cannot be loaded.

## Response sequence

1. Stop model or policy promotion.
2. Check whether the failure is model, application, dependency, or cluster related.
3. Scale replicas only when saturation is confirmed.
4. Route traffic to the last approved image when a release introduced the failure.
5. Preserve logs, metrics, release manifest, and audit identifiers for review.
6. Reopen promotion only after the incident evidence and corrective checks are complete.
