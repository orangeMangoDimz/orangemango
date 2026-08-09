# GitHub Release Pipeline Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize the single tag-triggered release workflow for faster execution and safer VPS deployments.

**Architecture:** Keep one `.github/workflows/release.yml` with separate quality, build, deploy, and cleanup jobs. Add release serialization, bounded execution, lean checkouts, a reusable SSH configuration, and a single Compose pull/start operation while preserving GHCR tagging and retention.

**Tech Stack:** GitHub Actions, Docker Buildx, GHCR, Docker Compose, OpenSSH, uv, Ruff.

---

### Task 1: Add workflow coordination and bounded jobs

**Files:**
- Modify: `.github/workflows/release.yml`

- [x] Add a workflow concurrency group named `orangemango-production-release` with `cancel-in-progress: false` so releases queue rather than deploy out of order.
- [x] Add `timeout-minutes: 10` to `quality`, `timeout-minutes: 20` to `build`, `timeout-minutes: 15` to `deploy`, and `timeout-minutes: 5` to `cleanup`.
- [x] Add `fetch-depth: 1` and `persist-credentials: false` to the quality and build checkouts.

### Task 2: Reduce deployment checkout and SSH repetition

**Files:**
- Modify: `.github/workflows/release.yml`

- [x] Configure the deploy checkout with `fetch-depth: 1`, `persist-credentials: false`, and sparse checkout paths for only `infra/docker/docker-compose.yml` and `infra/docker/docker-compose.prod.yml`.
- [x] Write an SSH config entry named `orangemango-vps` using the configured host, user, port, known-host file, and strict host verification.
- [x] Replace repeated SSH/SCP connection options with the `orangemango-vps` alias while keeping the GHCR token on SSH stdin.

### Task 3: Make the remote Compose rollout safer and less redundant

**Files:**
- Modify: `.github/workflows/release.yml`

- [x] Replace the separate `docker compose pull api` and `docker compose up` calls with `docker compose up -d --pull always --no-build --wait --wait-timeout 60 --remove-orphans`.
- [x] Keep the final `docker compose ps` output for deployment diagnostics.
- [x] Preserve the production override's `build: !reset null` so the VPS cannot build application source.

### Task 4: Verify the optimized workflow

**Files:**
- Verify: `.github/workflows/release.yml`
- Verify: `infra/docker/docker-compose.prod.yml`

- [x] Run `pre-commit run --all-files`.
- [x] Parse the workflow YAML and run `go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.7 .github/workflows/release.yml`.
- [x] Run `git diff --check` and confirm no files are staged.
- [x] Validate the merged production Compose configuration with a non-secret release tag.
- [x] Run Ruff checks and the existing Docker build; do not create test files.
