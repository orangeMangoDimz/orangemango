# GitHub Release Pipeline Design

**Date:** 2026-08-09

## Goal

Release the Orangemango API only from semantic-version Git tags. Each release is formatted, linted, built as a Docker image, pushed to GHCR, deployed to the VPS through SSH with Docker Compose, and followed by container-version cleanup.

## Architecture

GitHub-hosted Actions runners perform quality checks and build the image. The image is published to `ghcr.io`; the VPS never builds application code. The deployment job copies the base and production Compose files to the VPS, authenticates to GHCR, and runs Compose with the immutable release tag.

The production override changes only deployment concerns: it replaces the API `build` definition with an image reference, enables pull-before-start behavior, and enables API restart-on-failure. Runtime secrets remain in the VPS `.env` file and are never copied from GitHub Actions.

## Trigger and image tags

The workflow listens only to tags shaped like `vMAJOR.MINOR.PATCH`. The release tag is the deployment tag. Docker metadata also publishes the normalized semantic version, a commit SHA tag, and `latest`. Compose receives the exact Git tag through `IMAGE_TAG`, so deployment does not depend on the mutable `latest` tag.

## GitHub configuration

Non-sensitive configuration uses repository variables:

- `GHCR_IMAGE`: lowercase full image name, for example `ghcr.io/orangomangodimz/orangemango`.
- `GHCR_PACKAGE_NAME`: package name used by retention cleanup.
- `VPS_HOST`: `43.157.226.237`.
- `VPS_USER`: `ubuntu`.
- `VPS_SSH_PORT`: SSH port.
- `VPS_DEPLOY_PATH`: persistent Compose directory, for example `/opt/orangemango`.
- `IMAGE_VERSIONS_TO_KEEP`: number of recent GHCR versions to retain.

Secrets are limited to `VPS_SSH_PRIVATE_KEY`, `VPS_KNOWN_HOSTS`, `GHCR_PULL_USERNAME`, and `GHCR_PULL_TOKEN`. The workflow-generated `GITHUB_TOKEN` is used only for publishing and package cleanup permissions.

The VPS must already contain a production `.env` with the database, provider, and API authentication values. The workflow does not print or overwrite that file.

## Retention

The build uses GitHub Actions BuildKit cache to avoid duplicate build storage. After a successful deployment, the package cleanup step keeps the configured number of newest GHCR image versions and deletes older versions. The retention count is configurable through `IMAGE_VERSIONS_TO_KEEP`.

## Failure behavior

Formatting or lint failures stop the release before image publication. Image build or publication failures stop deployment. SSH, Compose validation, image pull, or Compose startup failures fail the deployment job. The deployed image remains the previous version until the new Compose startup succeeds.
