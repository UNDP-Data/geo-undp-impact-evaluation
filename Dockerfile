FROM ghcr.io/prefix-dev/pixi:bookworm-slim

# 1. Create the non-root user and ensure they own the base directory
RUN useradd -m -s /bin/bash undp && \
    mkdir -p /opt/undp-dashboard && \
    chown -R undp:undp /opt/undp-dashboard

WORKDIR /opt/undp-dashboard

# Install OS dependencies (must run as root)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# SWITCH TO NON-ROOT USER FOR APP SETUP
# ==========================================
USER undp

# 2. Install dependencies first for Docker caching
COPY --chown=undp:undp pixi.toml pixi.lock ./
# Temporarily hide the clean container environment
RUN pixi install --locked --frozen && mv .pixi .pixi_container

# 3. Copy application code (chowned so the undp user can read/write)
COPY --chown=undp:undp . .

# 4. BULLETPROOF: Delete the host's broken .pixi and restore the clean container version
RUN rm -rf /opt/undp-dashboard/.pixi && mv .pixi_container .pixi

# 5. Runtime workspace and config symlinks
RUN mkdir -p /opt/undp-dashboard/data /opt/undp-dashboard/outputs /opt/undp-dashboard/logos \
    && mkdir -p /home/undp/.undp_dashboard/workspaces \
    # NOTE: Intentionally removed the circular 'undp' symlink that broke the pipeline earlier \
    && cd /opt/undp-dashboard/config && \
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