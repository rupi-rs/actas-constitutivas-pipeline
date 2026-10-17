# syntax=docker/dockerfile:1

# Corporate Document Automation pipeline image.
#
# OpenShift runs containers under the restricted-v2 SCC, which assigns an
# arbitrary non-root UID that belongs to GID 0 (root group). The image therefore:
#   - runs as UID 1001 by default (non-root) but tolerates any UID in GID 0;
#   - makes /app group-writable for GID 0 (chgrp -R 0 and chmod -R g=u), so the
#     assigned UID can read the code and write data/, outputs/ and logs/;
#   - sets HOME=/tmp so tools that write to $HOME have a writable location.

# ---------------------------------------------------------------------------
# Builder stage: resolve and install locked dependencies with uv.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Runtime stage: minimal surface, non-root by default, GID 0 writable.
# ---------------------------------------------------------------------------
FROM registry.access.redhat.com/ubi9/python-312

LABEL org.opencontainers.image.title="actas-pipeline" \
      org.opencontainers.image.description="Corporate Document Automation pipeline for Mexican incorporation deeds (Actas Constitutivas)" \
      org.opencontainers.image.vendor="Corporate Document Automation" \
      org.opencontainers.image.version="1.0.0"

WORKDIR /app

COPY --from=builder /app /app

USER 0

RUN mkdir -p /app/data /app/outputs /app/logs /app/study_guides \
 && chgrp -R 0 /app \
 && chmod -R g=u /app

ENV HOME=/tmp \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER 1001

ENTRYPOINT ["actas-cli"]
CMD ["demo"]
