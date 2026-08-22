from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT: Path = Path(__file__).resolve().parents[1]
SOURCE_ROOTS: tuple[Path, ...] = (
    ROOT / "app",
    ROOT / "studio",
    ROOT / "scripts",
    ROOT / "main.py",
)
DISALLOWED_MODULES: frozenset[str] = frozenset({"re", "regex"})
DISALLOWED_SCHEMA_ARGUMENTS: frozenset[str] = frozenset({"pattern", "regex"})


def source_files() -> list[Path]:
    files: list[Path] = []
    for source in SOURCE_ROOTS:
        if source.is_file():
            files.append(source)
            continue
        files.extend(
            path for path in source.rglob("*.py") if "__pycache__" not in path.parts
        )
    return sorted(files)


def files_to_check() -> list[Path]:
    supplied: list[Path] = [Path(value) for value in sys.argv[1:]]
    if supplied:
        return [path for path in supplied if path.suffix == ".py"]
    return source_files()


def imported_module(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.ImportFrom):
        return str(node.module or "").partition(".")[0]
    return ""


def violations(path: Path) -> list[str]:
    tree: ast.AST = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module: str = alias.name.partition(".")[0]
                if module in DISALLOWED_MODULES:
                    found.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: import {module}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = imported_module(node)
            if module in DISALLOWED_MODULES:
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}: from {module}")
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in DISALLOWED_SCHEMA_ARGUMENTS:
                    found.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: "
                        f"schema argument {keyword.arg}"
                    )
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "__import__"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in DISALLOWED_MODULES
            ):
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}: dynamic import")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in DISALLOWED_MODULES
            ):
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}: dynamic import")
    return found


def main() -> int:
    found: list[str] = []
    for path in files_to_check():
        found.extend(violations(path))
    if not found:
        print("No prohibited regular-expression usage found.")
        return 0
    print("\n".join(found))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
