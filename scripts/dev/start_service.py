#!/usr/bin/env python3

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

subprocess.run(
    [
        "docker",
        "compose",
        "--env-file",
        ".env",
        "-f",
        "infra/docker/docker-compose.yml",
        "up",
        "--build",
        "-d",
    ],
    cwd=ROOT,
    check=True,
)
