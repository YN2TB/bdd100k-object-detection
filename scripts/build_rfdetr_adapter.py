"""Create and validate the isolated RF-DETR COCO dataset view."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.rfdetr import (  # noqa: E402
    DEFAULT_ADAPTER,
    DEFAULT_SOURCE,
    RFDETRAdapterError,
    build_adapter,
    validate_adapter,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--copy-mode", choices=("auto", "hardlink", "copy"), default="auto")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        report = (
            validate_adapter(args.source, args.out).to_dict()
            if args.validate_only
            else build_adapter(args.source, args.out, args.copy_mode)
        )
    except RFDETRAdapterError as exc:
        print(f"RF-DETR adapter: FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
