#!/usr/bin/env python3
"""Build and start the Orangemango Docker Compose services."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

MAX_PORT = 65535
MIN_PORT = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the launcher's command-line options."""
    parser = argparse.ArgumentParser(
        description="Build and start the Orangemango Docker services.",
        usage="%(prog)s [-y]",
    )
    parser.add_argument(
        "-y",
        dest="skip_confirmation",
        action="store_true",
        help="start without asking for confirmation",
    )
    return parser.parse_args(argv)


def compose_command() -> list[str]:
    """Return the shared Docker Compose command and project files."""
    return [
        "docker",
        "compose",
        "--env-file",
        ".env",
        "-f",
        "infra/docker/docker-compose.yml",
    ]


def resolve_api_port(project_root: Path, command: Sequence[str]) -> int:
    """Resolve and validate the published API port from Compose."""
    try:
        result = subprocess.run(
            [*command, "config", "--format", "json"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Docker Compose is required to start the services.") from exc
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            "Docker Compose could not resolve the project configuration."
        ) from exc

    try:
        configuration: dict[str, Any] = json.loads(result.stdout)
        published_port = configuration["services"]["api"]["ports"][0]["published"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Docker Compose returned no published API port.") from exc

    try:
        api_port = int(published_port)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid API_PORT: {published_port}") from exc

    if not MIN_PORT <= api_port <= MAX_PORT:
        raise RuntimeError(f"Invalid API_PORT: {api_port}")
    return api_port


def print_service_summary(api_port: int) -> None:
    """Print the services and their configured ports."""
    print("This will build and start these Docker services in the background:")
    print(f"  api      -> http://localhost:{api_port}")
    print("  postgres -> 127.0.0.1:5432 (or POSTGRES_PORT from .env)")


def confirm_start() -> bool:
    """Ask whether the services should be started."""
    try:
        confirmation = input("Continue? [y/N] ")
    except EOFError:
        confirmation = ""
    print()
    return confirmation.strip().lower() == "y"


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve configuration, confirm, and start the services."""
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    command = compose_command()

    try:
        api_port = resolve_api_port(project_root, command)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_service_summary(api_port)
    if not args.skip_confirmation and not confirm_start():
        print("Cancelled.")
        return 0

    completed = subprocess.run(
        [*command, "up", "--build", "-d"],
        cwd=project_root,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
