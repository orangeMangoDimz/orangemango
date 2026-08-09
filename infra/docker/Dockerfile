# [Base Image]
# Provides a small Python runtime for the API container.
FROM python:3.13-slim-trixie

# [Build Tools]
# Installs uv for dependency management during the image build.
COPY --from=ghcr.io/astral-sh/uv:0.11.30 /uv /uvx /bin/

# [Application Setup]
# Sets the working directory used by subsequent build and runtime commands.
WORKDIR /app

# [Python Runtime]
# Configures Python and uv for predictable container behavior.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# [Dependencies]
# Installs locked production dependencies without installing the project yet.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# [Application Code]
# Copies the API, ORM, migration, and LangGraph application packages.
COPY main.py ./
COPY app ./app
COPY studio ./studio
COPY scripts ./scripts
COPY alembic.ini ./

# [Runtime User]
# Runs the API as a non-root user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

# [Runtime Configuration]
# Adds the project virtual environment to the runtime PATH.
ENV PATH="/app/.venv/bin:$PATH"

# [Network]
# Documents the port exposed by the API container.
EXPOSE 8000

# [Startup]
# Applies database migrations before starting the API server.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn main:app --host 0.0.0.0 --port 8000"]
