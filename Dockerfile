# syntax=docker/dockerfile:1

# OpenShift restricted-v2 SCC assigns an arbitrary non-root UID in GID 0, so /app is group-writable (chgrp 0, g=u), HOME=/tmp and USER 1001.

# Builder stage: resolve and install locked dependencies with uv.
FROM registry.access.redhat.com/ubi9/python-312 AS builder

USER 0

ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

WORKDIR /app

# Dependency layer first, so source changes do not invalidate the install cache.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# Application source and remaining project files (filtered by .dockerignore).
COPY . .
RUN uv sync --frozen --no-dev

# Runtime stage: minimal surface, non-root by default, GID 0 writable.
FROM registry.access.redhat.com/ubi9/python-312

LABEL org.opencontainers.image.title="actas-pipeline" \
      org.opencontainers.image.description="Corporate Document Automation pipeline for Mexican incorporation deeds (Actas Constitutivas)" \
      org.opencontainers.image.vendor="Corporate Document Automation" \
      org.opencontainers.image.version="1.0.0"

WORKDIR /app

COPY --from=builder /app /app

USER 0

RUN mkdir -p /app/data /app/outputs /app/logs \
 && chgrp -R 0 /app \
 && chmod -R g=u /app

ENV HOME=/tmp \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER 1001

ENTRYPOINT ["actas-cli"]
CMD ["demo"]
