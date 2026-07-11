.PHONY: data features train reports contract batch load deployment incident openapi manifest approval site test lint security audit audit-registry sbom serve docker release-consistency evidence ci all registry-up registry-down registry-register registry-promote registry-rollback registry-status registry-verify registry-smoke

PLATFORM_VERSION := $(strip $(shell cat VERSION))
MODEL_VERSION ?= $(PLATFORM_VERSION)

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
contract:
	python -m src.governance.model_contract --metadata models/metadata.json --output-json models/model_contract.json --output-md docs/model_contract.md
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
	ruff check src tests scripts
security:
	bandit -q -r src scripts -lll
audit:
	python -m pip_audit -r requirements-runtime.lock --strict
audit-registry:
	python -m pip_audit -r requirements-mlflow.lock --strict
serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
docker:
	docker build -f docker/Dockerfile -t regulated-ai-mlops-platform:$(PLATFORM_VERSION) .
release-consistency:
	python scripts/check_release_consistency.py
registry-up:
	docker compose -f docker-compose.registry.yml up -d --build postgres minio minio-init mlflow
registry-down:
	docker compose -f docker-compose.registry.yml down -v --remove-orphans
registry-register:
	docker compose -f docker-compose.registry.yml --profile registry run --rm registry-cli register --model-version $(MODEL_VERSION)
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
evidence: release-consistency data features train reports contract batch load deployment incident openapi sbom manifest approval site test
ci: evidence lint security audit audit-registry
all: evidence
