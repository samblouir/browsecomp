#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


def main() -> None:
    parser = argparse.ArgumentParser(description="Decrypt a private .json.fernet transcript")
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    key = os.environ.get("BC250_ARTIFACT_FERNET_KEY")
    if not key:
        raise SystemExit("BC250_ARTIFACT_FERNET_KEY is not set")
    try:
        plaintext = Fernet(key.encode("ascii")).decrypt(args.path.read_bytes())
    except (ValueError, InvalidToken) as exc:
        raise SystemExit(f"Unable to decrypt transcript: {exc}") from exc
    if args.output:
        args.output.write_bytes(plaintext)
    else:
        print(plaintext.decode("utf-8"))


if __name__ == "__main__":
    main()
