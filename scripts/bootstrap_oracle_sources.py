#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from browsecomp250.browser import PageFetcher
from browsecomp250.config import load_config
from browsecomp250.guided_training import compile_guided_steps, controller_label_leaks
from browsecomp250.oracle_sources import discover_redacted_public_sources
from browsecomp250.search import create_search_provider
from browsecomp250.util import atomic_write_json, utc_now_iso
from scripts.run_full_guided_training import load_guide_records, parse_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Privately locate answer-redacted public sources for route-only training rows."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/star-headline.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--guide-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-queries", type=int, default=12)
    parser.add_argument("--max-sources", type=int, default=5)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_file.exists():
        load_dotenv(args.env_file, override=False)
    config = load_config(args.config)
    config.search.provider = "brave"
    config.search.live_preflight = False
    route_records, oracle_records = load_guide_records(args.guide_root.resolve())
    indices = parse_indices(args.indices, len(oracle_records))
    if args.limit is not None:
        indices = indices[: args.limit]
    indices = [
        index
        for index in indices
        if not (oracle_records[index].get("oracle") or {}).get("evidence_sources")
    ]
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    search = create_search_provider(config.search)
    browser = PageFetcher(config.browser)
    rows: dict[str, Any] = {}

    async def process(index: int) -> None:
        async with semaphore:
            record = oracle_records[index]
            sources, metrics = await discover_redacted_public_sources(
                record,
                search=search,
                browser=browser,
                max_queries=args.max_queries,
                max_sources=args.max_sources,
            )
            candidate = json.loads(json.dumps(record))
            candidate.setdefault("oracle", {})["evidence_sources"] = sources
            steps, review = compile_guided_steps(route_records[index], candidate)
            leaks = controller_label_leaks(
                question=str(record["item"]["question_text"]),
                oracle_record=record,
                steps=steps,
                review_guidance=review,
            )
            if leaks:
                raise RuntimeError(f"row {index} bootstrap leaked private aliases: {leaks}")
            item_id = str(record["item"]["item_id"])
            rows[item_id] = {
                "row_index": index,
                "sources": sources,
                "metrics": metrics,
            }
            print(
                json.dumps(
                    {
                        "event": "source_bootstrap_completed",
                        "row_index": index,
                        "source_count": len(sources),
                        "metrics": metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    try:
        await asyncio.gather(*(process(index) for index in indices))
    finally:
        await browser.close()
        await search.close()
    payload = {
        "schema_version": "1.0",
        "created_at": utc_now_iso(),
        "answer_strings_or_private_queries_persisted": False,
        "selected_rows": len(indices),
        "rows_with_sources": sum(bool(value["sources"]) for value in rows.values()),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    os.chmod(args.output, 0o600)
    return payload


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run(args))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "selected_rows": payload["selected_rows"],
                "rows_with_sources": payload["rows_with_sources"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
