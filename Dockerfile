# syntax=docker/dockerfile:1
FROM ghcr.io/prefix-dev/pixi:latest AS builder

WORKDIR /app
COPY pixi.toml pixi.lock ./
RUN --mount=type=cache,target=/root/.cache/rattler/cache \
    pixi install --locked --frozen

# Generate the activation wrapper containing PROJ_DATA, GDAL_DATA, etc.
RUN pixi shell-hook > /app/entrypoint.sh && \
    echo 'exec "$@"' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# =====================================================================
FROM debian:bookworm-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=builder /app/entrypoint.sh /entrypoint.sh
COPY . /app

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["streamlit", "run", "app.py"]