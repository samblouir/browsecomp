#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from browsecomp250.campaign import write_campaign_ledgers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build immutable first-pass and latest-repair BrowseComp ledgers."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/campaign-ledger"))
    args = parser.parse_args()
    print(json.dumps(write_campaign_ledgers(args.runs_root, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
