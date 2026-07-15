from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import __version__
from .browser import PageFetcher
from .config import AppConfig, load_config
from .constants import (
    DEFAULT_SUBSET_PATH,
    OFFICIAL_DATASET_ROWS,
    SUBSET_INDICES_SHA256,
    SUBSET_SIZE,
)
from .dataset import (
    dataset_path,
    download_dataset,
    validate_dataset_file,
    write_dataset_manifest,
)
from .llm import OpenAICompatibleClient, parse_json_action, settings_from_model_config
from .llm.protocol import action_from_tool_call
from .llm.tools import tool_schemas
from .report import paired_compare, sanitize_run, scan_public_tree, write_reports
from .run import BenchmarkEngine, RunStorage
from .search import create_search_provider
from .subset import load_indices, reference_indices
from .util import canonical_sha256

app = typer.Typer(
    no_args_is_help=True,
    help="Reproducible BrowseComp-250 evaluation runner for OpenAI-compatible APIs.",
)
console = Console()

ConfigPath = Annotated[Path, typer.Option("--config", "-c", help="YAML configuration file")]
EnvPath = Annotated[
    Path | None,
    typer.Option("--env-file", help="Optional dotenv file loaded before config expansion"),
]


def _load(config_path: Path, env_file: Path | None) -> AppConfig:
    if env_file:
        if not env_file.exists():
            raise typer.BadParameter(f"Environment file does not exist: {env_file}")
        load_dotenv(env_file, override=True)
    elif Path(".env").exists():
        load_dotenv(".env", override=False)
    return load_config(config_path)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            help="Print version and exit",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    del version


@app.command("print-config")
def print_config(
    config: ConfigPath = Path("configs/headline.yaml"), env_file: EnvPath = None
) -> None:
    cfg = _load(config, env_file)
    console.print_json(json.dumps(cfg.public_dict()))


@app.command()
def subset(
    path: Annotated[Path, typer.Option(help="Frozen subset index file")] = DEFAULT_SUBSET_PATH,
) -> None:
    indices = load_indices(path.resolve())
    table = Table(title="BrowseComp-250 subset")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("Size", str(len(indices)))
    table.add_row("Seed", "0")
    table.add_row("Source rows", str(OFFICIAL_DATASET_ROWS))
    table.add_row("Indices SHA-256", canonical_sha256(indices))
    table.add_row("Matches reference", str(indices == reference_indices()))
    table.add_row("First 20 indices", ", ".join(map(str, indices[:20])))
    console.print(table)


@app.command()
def prepare(
    config: ConfigPath = Path("configs/headline.yaml"),
    env_file: EnvPath = None,
    force: Annotated[bool, typer.Option(help="Redownload the official encrypted CSV")] = False,
) -> None:
    """Download and validate the encrypted official dataset; never writes plaintext items."""
    cfg = _load(config, env_file)

    async def action() -> tuple[Path, Path]:
        path = await download_dataset(cfg.dataset, force=force)
        manifest = write_dataset_manifest(cfg.dataset)
        return path, manifest

    path, manifest = asyncio.run(action())
    metadata = validate_dataset_file(path, cfg.dataset)
    console.print(f"[green]Dataset ready:[/green] {path}")
    console.print(f"Rows: {metadata['rows']}")
    console.print(f"SHA-256: [bold]{metadata['sha256']}[/bold]")
    console.print(f"Manifest: {manifest}")
    if not cfg.dataset.expected_sha256:
        console.print(
            "\nPin this value before a headline run:\n"
            f"[bold]BC250_EXPECTED_DATASET_SHA256={metadata['sha256']}[/bold]"
        )


@app.command()
def doctor(
    config: ConfigPath = Path("configs/smoke.yaml"),
    env_file: EnvPath = None,
    live: Annotated[
        bool,
        typer.Option(help="Also call the model, search provider, and a public webpage"),
    ] = False,
) -> None:
    cfg = _load(config, env_file)
    checks: list[tuple[str, bool, str]] = []
    try:
        indices = load_indices(cfg.dataset.subset_indices_path)
        checks.append(
            ("Frozen subset", True, f"{len(indices)} indices; {SUBSET_INDICES_SHA256[:12]}…")
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(("Frozen subset", False, str(exc)))

    path = dataset_path(cfg.dataset)
    if path.exists():
        try:
            meta = validate_dataset_file(path, cfg.dataset)
            checks.append(("Dataset", True, f"{meta['rows']} rows; {meta['sha256'][:12]}…"))
        except Exception as exc:  # noqa: BLE001
            checks.append(("Dataset", False, str(exc)))
    else:
        checks.append(("Dataset", False, f"Not downloaded; run `bc250 prepare -c {config}`"))

    checks.append(("Python", sys.version_info >= (3, 12), sys.version.split()[0]))
    model_auth_ok = bool(cfg.model.api_key) or cfg.model.allow_empty_api_key
    checks.append(
        (
            "Model API authentication",
            model_auth_ok,
            "bearer key present"
            if cfg.model.api_key
            else ("explicitly allowed without a key" if cfg.model.allow_empty_api_key else "empty"),
        )
    )
    checks.append(
        (
            "Search credentials",
            bool(cfg.search.selected_api_key()) or cfg.search.provider == "searxng",
            cfg.search.provider,
        )
    )
    checks.append(
        (
            "Grader credentials",
            cfg.grader.mode == "deterministic"
            or bool(cfg.grader.api_key)
            or cfg.grader.allow_empty_api_key,
            cfg.grader.mode,
        )
    )

    if live:

        async def live_checks() -> list[tuple[str, bool, str]]:
            output: list[tuple[str, bool, str]] = []
            model = OpenAICompatibleClient(settings_from_model_config(cfg.model))
            search = create_search_provider(cfg.search)
            browser = PageFetcher(cfg.browser)
            try:
                model_messages = [
                    {"role": "system", "content": "Use the supplied action protocol."},
                    {"role": "user", "content": "Save a note containing exactly: preflight"},
                ]
                if cfg.model.protocol in {"tools", "auto"}:
                    response = await model.chat(
                        model_messages,
                        tools=tool_schemas(),
                        tool_choice="auto",
                    )
                else:
                    response = await model.chat(model_messages)
                try:
                    calls = response.raw_message.get("tool_calls") or []
                    if cfg.model.protocol in {"tools", "auto"} and calls:
                        action = action_from_tool_call(calls[0])
                    else:
                        action = parse_json_action(response.content)
                    output.append(
                        (
                            "Model live call",
                            True,
                            f"{response.response_model or cfg.model.model}; action={action.action}",
                        )
                    )
                except Exception:
                    output.append(
                        ("Model live call", True, "responded; JSON-action format needs tuning")
                    )
            except Exception as exc:  # noqa: BLE001
                output.append(("Model live call", False, str(exc)))
            try:
                results = await search.search("OpenAI BrowseComp benchmark", count=3)
                output.append(("Search live call", bool(results), f"{len(results)} results"))
            except Exception as exc:  # noqa: BLE001
                output.append(("Search live call", False, str(exc)))
            try:
                document = await browser.fetch("https://example.com/")
                output.append(
                    ("Browser live call", len(document.text) > 100, f"{len(document.text)} chars")
                )
            except Exception as exc:  # noqa: BLE001
                output.append(("Browser live call", False, str(exc)))
            finally:
                await model.close()
                await search.close()
                await browser.close()
            return output

        checks.extend(asyncio.run(live_checks()))

    table = Table(title="BrowseComp-250 doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for name, ok, detail in checks:
        table.add_row(name, "[green]PASS[/green]" if ok else "[red]FAIL[/red]", detail)
    console.print(table)
    if any(not ok for _, ok, _ in checks):
        raise typer.Exit(1)


@app.command()
def run(
    config: ConfigPath = Path("configs/smoke.yaml"),
    env_file: EnvPath = None,
    limit: Annotated[int | None, typer.Option(help="Run the first N frozen subset items")] = None,
) -> None:
    cfg = _load(config, env_file)
    if not dataset_path(cfg.dataset).exists():
        console.print("Dataset is missing; downloading the encrypted official CSV.")
        asyncio.run(download_dataset(cfg.dataset))
    summary = asyncio.run(BenchmarkEngine(cfg).run(limit=limit))
    console.print_json(json.dumps(summary))
    console.print(f"Run directory: [bold]{cfg.run.output_dir / cfg.run.name}[/bold]")


def _validate_headline(cfg: AppConfig, allow_unpinned_dataset: bool) -> list[str]:
    problems: list[str] = []
    if cfg.dataset.subset_size != SUBSET_SIZE:
        problems.append(f"subset_size must be {SUBSET_SIZE}")
    if load_indices(cfg.dataset.subset_indices_path) != reference_indices():
        problems.append("subset index file does not match the frozen seed-0 subset")
    if cfg.run.attempts != 1:
        problems.append("headline protocol requires exactly one attempt per item")
    if cfg.grader.mode != "official_llm":
        problems.append("headline protocol requires grader.mode=official_llm")
    if not cfg.dataset.expected_sha256 and not allow_unpinned_dataset:
        problems.append("dataset SHA-256 is unpinned; run `bc250 prepare` and set it")
    minimums = {
        "max_steps": (cfg.agent.max_steps, 60),
        "max_search_calls": (cfg.agent.max_search_calls, 30),
        "max_page_opens": (cfg.agent.max_page_opens, 60),
        "task_timeout_seconds": (cfg.run.task_timeout_seconds, 900),
    }
    for name, (actual, minimum) in minimums.items():
        if actual < minimum:
            problems.append(f"{name}={actual} is below headline minimum {minimum}")
    if not cfg.model.api_key and not cfg.model.allow_empty_api_key:
        problems.append(
            "model API key is empty; set a key or explicitly set allow_empty_api_key=true"
        )
    if not cfg.grader.api_key and not cfg.grader.allow_empty_api_key:
        problems.append(
            "grader API key is empty; set a key or explicitly set grader.allow_empty_api_key=true"
        )
    if cfg.search.provider != "searxng" and not cfg.search.selected_api_key():
        problems.append(f"{cfg.search.provider} search API key is empty")
    return problems


@app.command()
def headline(
    config: ConfigPath = Path("configs/headline.yaml"),
    env_file: EnvPath = None,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm the complete 250-item run")] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Validate and print the locked protocol only")
    ] = False,
    allow_unpinned_dataset: Annotated[
        bool,
        typer.Option(help="Permit a run without a configured source CSV SHA-256"),
    ] = False,
) -> None:
    cfg = _load(config, env_file)
    problems = _validate_headline(cfg, allow_unpinned_dataset)
    if problems:
        console.print("[red]Headline protocol validation failed:[/red]")
        for problem in problems:
            console.print(f"- {problem}")
        raise typer.Exit(2)
    console.print_json(json.dumps(cfg.public_dict()))
    console.print(f"Frozen subset SHA-256: {SUBSET_INDICES_SHA256}")
    console.print(f"Planned trials: {SUBSET_SIZE}")
    if dry_run:
        return
    if not yes:
        console.print("Pass --yes to launch the full evaluation.")
        raise typer.Exit(2)
    if not dataset_path(cfg.dataset).exists():
        asyncio.run(download_dataset(cfg.dataset))
    summary = asyncio.run(BenchmarkEngine(cfg).run(limit=None))
    console.print_json(json.dumps(summary))


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="Completed run directory")],
    confidence: Annotated[float, typer.Option(help="Confidence level")] = 0.95,
    bootstrap_samples: Annotated[int, typer.Option(help="Bootstrap replicates")] = 10_000,
) -> None:
    storage = RunStorage(run_dir.resolve())
    records = storage.load_records()
    summary = write_reports(
        run_dir.resolve(),
        records,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        write_csv=True,
        write_html=True,
    )
    console.print_json(json.dumps(summary))


@app.command()
def sanitize(
    run_dir: Annotated[Path, typer.Argument(help="Completed run directory")],
    destination: Annotated[Path, typer.Argument(help="Publication-safe output directory")],
    config: ConfigPath = Path("configs/headline.yaml"),
    env_file: EnvPath = None,
) -> None:
    cfg = _load(config, env_file)
    output = sanitize_run(run_dir.resolve(), destination.resolve(), cfg)
    console.print(f"Publication-safe artifacts: [green]{output}[/green]")


@app.command("verify-run")
def verify_run(
    run_dir: Annotated[Path, typer.Argument(help="Run directory")],
    config: ConfigPath = Path("configs/headline.yaml"),
    env_file: EnvPath = None,
) -> None:
    cfg = _load(config, env_file)
    run_dir = run_dir.resolve()
    errors: list[str] = []
    lock_path = run_dir / "run.lock.json"
    if not lock_path.exists():
        errors.append("missing run.lock.json")
    else:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("subset_indices_sha256") != SUBSET_INDICES_SHA256:
            errors.append("subset hash mismatch in run lock")
    try:
        errors.extend(scan_public_tree(run_dir, cfg))
    except FileNotFoundError as exc:
        errors.append(str(exc))
    storage = RunStorage(run_dir)
    try:
        records = storage.load_records()
        if not records:
            errors.append("no trial records")
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    if errors:
        console.print("[red]Run verification failed:[/red]")
        for error in errors:
            console.print(f"- {error}")
        raise typer.Exit(1)
    console.print("[green]Run verification passed.[/green]")


@app.command()
def compare(
    run_dirs: Annotated[list[Path], typer.Argument(help="Two or more run directories")],
) -> None:
    table = Table(title="BrowseComp-250 comparison")
    table.add_column("Run")
    table.add_column("Model")
    table.add_column("Accuracy")
    table.add_column("95% CI")
    table.add_column("Cost")
    table.add_column("Median sec")
    for run_dir in run_dirs:
        summary_path = run_dir / "public" / "summary.json"
        lock_path = run_dir / "run.lock.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        model = lock.get("config", {}).get("model", {}).get("model", "unknown")
        low, high = summary["wilson_interval"]
        table.add_row(
            run_dir.name,
            str(model),
            f"{100 * summary['accuracy']:.2f}%",
            f"{100 * low:.2f}–{100 * high:.2f}%",
            f"${summary['cost_usd']:.2f}",
            f"{summary['duration_seconds']['median'] or 0:.1f}",
        )
    console.print(table)


@app.command("paired-compare")
def paired_compare_command(
    left_run: Annotated[Path, typer.Argument(help="First run directory")],
    right_run: Annotated[Path, typer.Argument(help="Second run directory")],
    bootstrap_samples: Annotated[int, typer.Option(help="Paired bootstrap replicates")] = 10_000,
    confidence: Annotated[float, typer.Option(help="Confidence level")] = 0.95,
    output: Annotated[Path | None, typer.Option(help="Optional JSON output path")] = None,
) -> None:
    result = paired_compare(
        left_run.resolve(),
        right_run.resolve(),
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    console.print_json(json.dumps(result))
    if result["protocol_mismatches"]:
        console.print(
            "[yellow]Protocol mismatch warning:[/yellow] "
            + ", ".join(result["protocol_mismatches"])
        )


if __name__ == "__main__":
    app()
