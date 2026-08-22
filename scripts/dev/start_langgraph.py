#!/usr/bin/env python3

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]

subprocess.run(
    ["uv", "run", "langgraph", "dev", "--config", "langgraph.json"],
    cwd=ROOT / "app",
    check=True,
)
