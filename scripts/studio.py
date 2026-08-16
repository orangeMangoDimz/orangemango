#!/usr/bin/env python3
"""Start the local LangGraph Studio server."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

ALL_GRAPHS = (
    "cv-extraction",
    "job-extraction",
    "matching-score",
    "orchestrator",
    "cv-job-chatbot",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the launcher's command-line options."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_args:
        separator = raw_args.index("--")
        graph_args, langgraph_args = raw_args[:separator], raw_args[separator + 1 :]
    else:
        graph_args, langgraph_args = raw_args, []

    parser = argparse.ArgumentParser(
        description="Start the local LangGraph Studio server.",
        usage="%(prog)s [GRAPH ...] [-- [LANGGRAPH_OPTIONS ...]]",
    )
    parser.add_argument(
        "graphs",
        nargs="*",
        metavar="GRAPH",
        choices=ALL_GRAPHS,
        help="graphs to run (default: all)",
    )
    args = parser.parse_args(graph_args)
    args.langgraph_args = langgraph_args
    return args


def write_selected_config(
    config_path: Path, selected_graphs: Sequence[str]
) -> Path:
    """Write a temporary Studio config containing only the selected graphs."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["graphs"] = {
        name: path
        for name, path in config["graphs"].items()
        if name in selected_graphs
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=config_path.parent,
        prefix=".langgraph-selected.",
        suffix=".json",
        delete=False,
    ) as handle:
        handle.write(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        return Path(handle.name)


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve the graphs to run and start the Studio server."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    studio_dir = project_root / "studio"
    config_path = studio_dir / "langgraph.json"

    if not config_path.is_file():
        print(f"Error: Studio config not found: {config_path}", file=sys.stderr)
        return 1

    selected_graphs = list(dict.fromkeys(args.graphs)) or list(ALL_GRAPHS)

    config_to_use = config_path
    temporary_config: Path | None = None
    if len(selected_graphs) != len(ALL_GRAPHS):
        temporary_config = write_selected_config(config_path, selected_graphs)
        config_to_use = temporary_config

    print(f"Starting LangGraph Studio with: {' '.join(selected_graphs)}")
    try:
        completed = subprocess.run(
            [
                "uv",
                "run",
                "langgraph",
                "dev",
                "--config",
                str(config_to_use),
                *args.langgraph_args,
            ],
            cwd=studio_dir,
            check=False,
        )
        return completed.returncode
    except FileNotFoundError:
        print("Error: uv is required to start the Studio server.", file=sys.stderr)
        return 1
    finally:
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
