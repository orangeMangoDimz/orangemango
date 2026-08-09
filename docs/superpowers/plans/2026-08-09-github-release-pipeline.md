# GitHub Release Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tag-only GitHub Actions release pipeline that checks code quality, publishes a versioned GHCR image, deploys it to the VPS with Docker Compose, and removes old image versions.

**Architecture:** GitHub-hosted runners run Ruff and BuildKit, publish to GHCR, and connect over SSH to the persistent VPS. The VPS keeps its own `.env`; a production Compose override selects the immutable GHCR tag and does not build locally.

**Tech Stack:** GitHub Actions, uv, Ruff, Docker Buildx, GHCR, Docker Compose, OpenSSH.

---

### Task 1: Add reproducible formatting and lint tooling

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [x] Add Ruff to the existing `dev` dependency group without changing production dependencies.
- [x] Regenerate the lockfile with `uv lock`.
- [x] Run `uv run ruff format --check .` and `uv run ruff check .`; if existing source violations are reported, apply only Ruff formatting/lint fixes required for the release gate and rerun both commands.

### Task 2: Add the production Compose override

**Files:**
- Create: `infra/docker/docker-compose.prod.yml`

- [x] Define an `api` override with `build: !reset null`, `image: ${GHCR_IMAGE:?GHCR_IMAGE must be set}:${IMAGE_TAG:?IMAGE_TAG must be set}`, `pull_policy: always`, and `restart: unless-stopped`.
- [x] Leave inherited services, runtime environment, database volume, and ports unchanged unless the merged configuration requires an override.
- [x] Validate the merged configuration with `docker compose --env-file .env -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.prod.yml config --quiet` using a non-secret temporary `IMAGE_TAG` and the existing local `.env`.

### Task 3: Add the tag-triggered release workflow

**Files:**
- Create: `.github/workflows/release.yml`

- [x] Trigger only on `push` tags matching `v*.*.*`, then reject tags that do not match `^v[0-9]+\\.[0-9]+\\.[0-9]+$`.
- [x] Use repository variables for image name, VPS address, SSH port, deployment path, package name, and retention count.
- [x] Use `actions/checkout`, `astral-sh/setup-uv`, Docker Buildx, GHCR login, Docker metadata, and Docker build/push actions with pinned versions or SHAs.
- [x] Run Ruff formatting and linting before the build job.
- [x] Publish the exact release tag, normalized semantic aliases, SHA, and `latest` while deploying the exact release tag.
- [x] Install the SSH key with known-host verification, copy both Compose files, log into GHCR on the VPS through stdin, and run Compose with `up -d --pull always --no-build --wait --remove-orphans`.
- [x] Run GHCR retention cleanup after a successful deployment, retaining the configured number of newest versions.

### Task 4: Verify the release configuration

**Files:**
- Verify: `.github/workflows/release.yml`
- Verify: `infra/docker/docker-compose.prod.yml`

- [x] Parse the workflow YAML with a YAML parser.
- [x] Run Ruff formatting and lint checks.
- [x] Run the merged Compose configuration check and a Docker build using the existing Dockerfile.
- [x] Confirm `.env` is ignored and no secret values are staged.
- [x] Confirm the final diff excludes the pre-existing `.gitignore`, README deletion, and database-schema deletion.
