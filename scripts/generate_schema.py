#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from browsecomp250.config import AppConfig

root = Path(__file__).resolve().parents[1]
destination = root / "configs" / "schema.json"
destination.write_text(json.dumps(AppConfig.model_json_schema(), indent=2) + "\n", encoding="utf-8")
print(destination)
