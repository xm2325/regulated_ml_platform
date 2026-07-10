from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.serving.app import app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/openapi.json")
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
