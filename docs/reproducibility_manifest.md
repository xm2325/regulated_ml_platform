# Reproducibility manifest

## Manifest status: **PASS**

Git commit: `unknown`  
Python: `3.13.5`  
Created: `2026-07-10T05:33:56.697901+00:00`

## Artifact checksums

| Path | Exists | Bytes | SHA-256 |
|---|---|---:|---|
| `data/raw/customers.csv` | True | 399098 | `810002b190dc2748...` |
| `data/processed/features.csv` | True | 745188 | `ff7e8eef04462d46...` |
| `models/model.joblib` | True | 4507586 | `3c612ef470d73871...` |
| `models/metadata.json` | True | 910 | `1cbf4f40443d5192...` |
| `models/model_contract.json` | True | 5361 | `73091b68811384cd...` |
| `reports/model_metrics.json` | True | 14653 | `87652f7e842fff04...` |
| `reports/promotion_gate.json` | True | 1026 | `503a8bef01db6a9c...` |
| `reports/data_quality_report.json` | True | 632 | `a14d3fe409513ac8...` |
| `reports/privacy_report.json` | True | 720 | `bc16913a32284ef8...` |
| `config/policy.yaml` | True | 319 | `156fb42a999b9fe3...` |
| `config/promotion_gate.yaml` | True | 261 | `949e49b59a7e1ced...` |

## Package versions

- `fastapi==0.128.2`
- `pydantic==2.13.4`
- `numpy==2.3.5`
- `pandas==2.2.3`
- `scikit-learn==1.8.0`
- `joblib==1.5.3`
- `prometheus-client==0.25.0`
- `PyYAML==6.0.3`
