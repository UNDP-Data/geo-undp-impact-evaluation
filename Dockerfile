FROM ghcr.io/prefix-dev/pixi:bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 1. Install dependencies first for Docker caching
COPY pixi.toml pixi.lock ./
# Temporarily hide the clean container environment
RUN pixi install --locked --frozen && mv .pixi .pixi_container

# 2. Copy application code (if .dockerignore fails, it copies the broken host .pixi here)
COPY . /app

# 3. BULLETPROOF: Delete the host's broken .pixi and restore the clean container version
RUN rm -rf /app/.pixi && mv .pixi_container .pixi

# 4. Runtime workspace and config symlinks
RUN mkdir -p /app/data /app/outputs /app/logos \
    && mkdir -p /root/.undp_dashboard/workspaces \
    && ln -sf /app /root/.undp_dashboard/workspaces/undp \
    && cd /app/config && \
       (ln -sf afghanistan.yaml afg.yaml 2>/dev/null || true) && \
       (ln -sf myanmar.yaml mmr.yaml 2>/dev/null || true) && \
       (ln -sf ukraine.yaml ukr.yaml 2>/dev/null || true) && \
       (ln -sf gaza.yaml pse.yaml 2>/dev/null || true)

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["pixi", "run"]
CMD ["streamlit", "run", "app.py"]