#!/usr/bin/env bash
set -euo pipefail

# Parse command-line options.
usage() {
  printf 'Usage: %s [-y]\n' "${0##*/}"
}

skip_confirmation=false
while (($# > 0)); do
  case "$1" in
    -y)
      skip_confirmation=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Error: unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Run Compose from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Show the services and ports before starting them.
printf '%s\n' \
  'This will build and start these Docker services in the background:' \
  '  api      -> http://localhost:8000' \
  '  postgres -> 127.0.0.1:5432 (or POSTGRES_PORT from .env)'

# Confirm before starting unless -y was provided.
if [[ "$skip_confirmation" != true ]]; then
  printf 'Continue? [y/N] '
  IFS= read -r confirmation || confirmation=''
  printf '\n'

  if [[ ! "$confirmation" =~ ^[Yy]$ ]]; then
    printf '%s\n' 'Cancelled.'
    exit 0
  fi
fi

# Build and start the services in the background.
exec docker compose --env-file .env -f infra/docker/docker-compose.yml up --build -d
