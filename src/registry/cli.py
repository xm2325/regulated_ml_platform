from __future__ import annotations

import argparse
import json

from src.registry.lifecycle import (
    RegistryConfig,
    load_json,
    promote_challenger,
    register_release,
    registry_status,
    rollback_champion,
    sync_alias,
    verify_alias,
    write_json,
)


def _config(args: argparse.Namespace) -> RegistryConfig:
    base = RegistryConfig.from_env()
    return RegistryConfig(
        tracking_uri=args.tracking_uri or base.tracking_uri,
        registry_uri=args.registry_uri or base.registry_uri,
        experiment_name=args.experiment_name or base.experiment_name,
        registered_model_name=args.model_name or base.registered_model_name,
        champion_alias=base.champion_alias,
        challenger_alias=base.challenger_alias,
        rollback_alias=base.rollback_alias,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled MLflow model-registry lifecycle")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--registry-uri")
    parser.add_argument("--experiment-name")
    parser.add_argument("--model-name")
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser("register")
    register.add_argument("--model", default="models/model.joblib")
    register.add_argument("--metadata", default="models/metadata.json")
    register.add_argument("--metrics", default="reports/model_metrics.json")
    register.add_argument("--gate", default="reports/promotion_gate.json")
    register.add_argument("--model-version")
    register.add_argument("--output", default="reports/registry_register.json")

    promote = commands.add_parser("promote")
    promote.add_argument("--gate", default="reports/promotion_gate.json")
    promote.add_argument("--expected-challenger-version")
    promote.add_argument("--output", default="reports/registry_promotion.json")

    rollback = commands.add_parser("rollback")
    rollback.add_argument("--reason", required=True)
    rollback.add_argument("--output", default="reports/registry_rollback.json")

    status = commands.add_parser("status")
    status.add_argument("--output", default="reports/registry_status.json")

    sync = commands.add_parser("sync")
    sync.add_argument("--alias", default="champion")
    sync.add_argument("--output-dir", default="registry-sync")
    sync.add_argument("--output", default="reports/registry_sync.json")

    verify = commands.add_parser("verify")
    verify.add_argument("--alias", default="champion")
    verify.add_argument("--request", default="examples/review_request.json")
    verify.add_argument("--output", default="reports/registry_verify.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _config(args)
    if args.command == "register":
        result = register_release(
            config, args.model, args.metadata, args.metrics, args.gate, args.model_version, args.output
        )
    elif args.command == "promote":
        result = promote_challenger(
            config,
            load_json(args.gate),
            expected_challenger_version=args.expected_challenger_version,
            output_path=args.output,
        )
    elif args.command == "rollback":
        result = rollback_champion(config, args.reason, output_path=args.output)
    elif args.command == "status":
        result = registry_status(config)
        write_json(args.output, result)
    elif args.command == "sync":
        result = sync_alias(config, args.alias, args.output_dir, output_path=args.output)
    else:
        result = verify_alias(config, args.alias, args.request, output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
