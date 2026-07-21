.PHONY: data features train reports continuous alerts contract triton-export triton-validate triton-parity triton-benchmark accelerator triton triton-runtime-cpu batch load deployment incident openapi manifest approval site test lint security audit audit-onnx audit-registry audit-registry-client sbom serve docker evidence ci all registry-up registry-down registry-register registry-promote registry-rollback registry-status registry-verify registry-smoke

data:
	python -m src.data.make_dataset --n 5000 --output data/raw/customers.csv --seed 42
features:
	python -m src.features.build_features --input data/raw/customers.csv --output data/processed/features.csv
train:
	python -m src.models.train --input data/processed/features.csv --model-dir models --reports-dir reports
reports:
	python -m src.monitoring.drift_report --reference data/processed/features.csv --current data/processed/features.csv --temporal-split --output reports/drift_report.html
	python -m src.monitoring.data_quality --input data/raw/customers.csv --output reports/data_quality_report.json
	python -m src.governance.privacy_guard --input data/raw/customers.csv --output-json reports/privacy_report.json --output-md reports/privacy_report.md
	python -m src.governance.generate_model_card --metrics reports/model_metrics.json --output docs/model_card.md
	python -m src.governance.generate_data_release_note --input data/processed/features.csv --output docs/data_release_note.md
	python -m src.governance.promotion_gate --metrics reports/model_metrics.json --config config/promotion_gate.yaml --output-json reports/promotion_gate.json --output-md reports/promotion_gate.md
	python -m src.models.champion_challenger --metrics reports/model_metrics.json --output-json reports/champion_challenger_report.json --output-md reports/champion_challenger_report.md
	python -m src.governance.explainability_report --model models/model.joblib --data data/processed/features.csv --output-json reports/explainability_report.json --output-md reports/explainability_report.md --sample-size 600
continuous:
	python -m src.operations.continuous_ops --drift reports/drift_summary.json --data-quality reports/data_quality_report.json --output-json reports/continuous_ops_decision.json --output-md reports/continuous_ops_decision.md
alerts:
	python -m src.operations.validate_alerting --rules observability/prometheus/regulated-ai-alerts.yaml --repo-root . --output reports/alerting_validation.json
contract:
	python -m src.governance.model_contract --metadata models/metadata.json --output-json models/model_contract.json --output-md docs/model_contract.md
triton-export:
	python -m src.serving.triton_export --model models/model.joblib --metadata models/metadata.json --sample data/processed/features.csv --output-root models/triton
	python -m src.serving.normalize_onnx_ir --model models/triton/model_repository/support_calibrator/1/model.onnx --contract models/triton/contract.json --artifact-key support_calibrator_onnx_sha256 --max-ir-version 10 --output reports/onnx_ir_compatibility.json
triton-validate:
	python -m src.operations.validate_triton_repository --root models/triton --output reports/triton_repository_validation.json
triton-parity:
	python -m src.serving.validate_triton_export --model models/model.joblib --sample data/processed/features.csv --triton-root models/triton --output reports/triton_onnx_parity.json
triton-benchmark:
	python -m src.operations.benchmark_onnx_cpu --model models/model.joblib --sample data/processed/features.csv --triton-root models/triton --output reports/onnx_cpu_benchmark.json
accelerator:
	python -m src.operations.accelerator_policy --contract models/triton/contract.json --policy config/accelerator_policy.yaml --output reports/accelerator_decision.json
triton: triton-export triton-validate triton-parity triton-benchmark accelerator
	python -m pip freeze | grep -E '^(onnx|onnxruntime|skl2onnx)==' > reports/onnx_toolchain_versions.txt
triton-runtime-cpu:
	bash scripts/triton_cpu_runtime_smoke.sh
batch:
	python -m src.serving.batch_score --input data/raw/customers.csv --output reports/batch_predictions.csv
load:
	python -m src.operations.load_test --requests 200 --output-json reports/load_test_summary.json --output-md reports/load_test_report.md
deployment:
	python -m src.operations.validate_deployment --root k8s --output-json reports/deployment_validation.json --output-md reports/deployment_validation.md
incident:
	python -m src.operations.incident_drill --reports-dir reports --output-json reports/incident_drill_report.json --output-md reports/incident_drill_report.md
openapi:
	python -m src.operations.export_openapi --output docs/openapi.json
sbom:
	python -m src.operations.generate_sbom --output reports/sbom.cdx.json
manifest:
	python -m src.governance.reproducibility_manifest --root . --output-json reports/reproducibility_manifest.json --output-md docs/reproducibility_manifest.md
approval:
	python -m src.governance.change_approval_pack --reports-dir reports --output-json reports/release_approval_pack.json --output-md docs/release_approval_pack.md
site:
	python -m src.operations.build_showcase --root . --output site
test:
	pytest -q
lint:
	ruff check src tests
security:
	bandit -q -r src -lll
audit:
	python -m pip_audit -r requirements-runtime.lock --strict
audit-onnx:
	python -m pip_audit -r requirements-onnx.txt --strict
audit-registry:
	python -m pip_audit -r requirements-mlflow.lock --strict
audit-registry-client:
	python -m pip_audit -r requirements-registry-client.lock --strict
serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
docker:
	docker build -f docker/Dockerfile -t regulated-ai-mlops-platform:1.1.0 .
registry-up:
	docker compose -f docker-compose.registry.yml up -d --build postgres minio minio-init mlflow
registry-down:
	docker compose -f docker-compose.registry.yml down -v --remove-orphans
registry-register:
	docker compose -f docker-compose.registry.yml --profile registry run --rm registry-cli register --model-version 0.6.0
registry-promote:
	docker compose -f docker-compose.registry.yml --profile registry run --rm registry-cli promote --gate reports/promotion_gate.json
registry-rollback:
	docker compose -f docker-compose.registry.yml --profile registry run --rm registry-cli rollback --reason "manual rollback"
registry-status:
	docker compose -f docker-compose.registry.yml --profile registry run --rm registry-cli status
registry-verify:
	docker compose -f docker-compose.registry.yml --profile registry run --rm registry-cli verify --alias champion --request examples/review_request.json
registry-smoke:
	bash scripts/registry_stack_smoke.sh
evidence: data features train triton reports continuous alerts contract batch load deployment incident openapi sbom manifest approval site test
ci: evidence lint security audit audit-onnx audit-registry audit-registry-client
all: evidence
