# GitHub Release Pipeline Optimization Design

**Date:** 2026-08-09

## Goal

Make the tag-triggered release workflow faster and safer without splitting it into multiple workflow files or changing its GHCR and VPS deployment contract.

## Design

Keep `.github/workflows/release.yml` as the single workflow, with four independent jobs connected by `needs`: quality, image build, deployment, and retention cleanup. This preserves least-privilege permissions and makes failures visible by stage.

The workflow serializes production releases with a static concurrency group and does not cancel an in-progress release. Each job has a timeout. Quality and build checkouts use shallow history and do not persist Git credentials; deployment uses sparse checkout because it only needs the two Compose files.

The deployment job configures one SSH alias with strict known-host verification, then reuses it for directory creation, file copy, GHCR login, and Compose execution. Compose pulls the configured API image, never builds on the VPS, waits for startup, and reports the final service state.

## Preserved behavior

- Only `vMAJOR.MINOR.PATCH` tag pushes trigger the workflow.
- The exact release tag is deployed from GHCR.
- BuildKit cache, provenance, SBOM, semantic aliases, SHA, `latest`, and GHCR retention remain enabled.
- The VPS `.env` remains server-side and is never overwritten by the workflow.

## Verification

Run pre-commit, YAML parsing, `actionlint`, `git diff --check`, the merged production Compose configuration check, and the existing local Docker build. Do not add test files for this workflow-only optimization.
