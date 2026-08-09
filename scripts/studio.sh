#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly STUDIO_DIR="${PROJECT_ROOT}/studio"
readonly CONFIG_PATH="${STUDIO_DIR}/langgraph.json"

readonly ALL_GRAPHS=(
  "cv-extraction"
  "job-extraction"
  "matching-score"
  "orchestrator"
  "cv-job-chatbot"
)

usage() {
  cat <<'EOF'
Usage: scripts/studio.sh [GRAPH ...] [-- [LANGGRAPH_OPTIONS ...]]

Start the local LangGraph Studio server.

GRAPH values:
  cv-extraction   Run the CV extraction graph
  job-extraction  Run the job extraction graph
  matching-score  Run the matching score graph
  orchestrator    Run the CV/job orchestrator graph
  cv-job-chatbot  Run the conversational CV/job chatbot graph

If no GRAPH values are supplied, all graphs are started.
Pass additional LangGraph CLI options after --, for example:
  scripts/studio.sh -- --port 2025 --no-browser
EOF
}

is_known_graph() {
  local candidate="$1"
  local graph

  for graph in "${ALL_GRAPHS[@]}"; do
    if [[ "$graph" == "$candidate" ]]; then
      return 0
    fi
  done

  return 1
}

if [[ ! -f "$CONFIG_PATH" ]]; then
  printf 'Error: Studio config not found: %s\n' "$CONFIG_PATH" >&2
  exit 1
fi

selected_graphs=()
langgraph_args=()
parsing_graphs=true

while (($# > 0)); do
  if [[ "$parsing_graphs" == true && "$1" == "--" ]]; then
    parsing_graphs=false
    shift
    continue
  fi

  if [[ "$parsing_graphs" == false ]]; then
    langgraph_args+=("$1")
    shift
    continue
  fi

  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -* )
      printf 'Error: CLI options must come after --: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if ! is_known_graph "$1"; then
        printf 'Error: unknown graph: %s\n' "$1" >&2
        usage >&2
        exit 2
      fi

      already_selected=false
      for selected in "${selected_graphs[@]}"; do
        if [[ "$selected" == "$1" ]]; then
          already_selected=true
          break
        fi
      done

      if [[ "$already_selected" == false ]]; then
        selected_graphs+=("$1")
      fi
      shift
      ;;
  esac
done

if ((${#selected_graphs[@]} == 0)); then
  selected_graphs=("${ALL_GRAPHS[@]}")
fi

config_to_use="$CONFIG_PATH"
temporary_config=''
cleanup() {
  if [[ -n "$temporary_config" && -f "$temporary_config" ]]; then
    rm -f -- "$temporary_config"
  fi
}
trap cleanup EXIT

if ((${#selected_graphs[@]} != ${#ALL_GRAPHS[@]})); then
  temporary_config="$(mktemp "${STUDIO_DIR}/.langgraph-selected.XXXXXX.json")"

  python3 - "$CONFIG_PATH" "$temporary_config" "${selected_graphs[@]}" <<'PY'
import json
import sys
from pathlib import Path


source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
selected_graphs = set(sys.argv[3:])

config = json.loads(source_path.read_text(encoding="utf-8"))
config["graphs"] = {
    name: path
    for name, path in config["graphs"].items()
    if name in selected_graphs
}
target_path.write_text(
    json.dumps(config, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

  config_to_use="$temporary_config"
fi

printf 'Starting LangGraph Studio with: %s\n' "${selected_graphs[*]}"
cd "$STUDIO_DIR"
uv run langgraph dev --config "$config_to_use" "${langgraph_args[@]}"
