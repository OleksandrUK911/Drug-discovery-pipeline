# ═══════════════════════════════════════════════════════════════
# Drug Discovery ML Portfolio — Streamlit Dashboard
# ═══════════════════════════════════════════════════════════════
# Build:  docker build -t drug-discovery-dashboard .
# Run:    docker run -p 8501:8501 drug-discovery-dashboard
# Or use: docker-compose up
# ═══════════════════════════════════════════════════════════════

# Stage 1: Base with conda + RDKit (rdkit can't be installed via pip reliably)
FROM continuumio/miniconda3:24.1.2-0 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONDA_AUTO_UPDATE_CONDA=false \
    PATH="/opt/conda/bin:$PATH"

# Create a working directory
WORKDIR /app

# ── Install RDKit + Python 3.11 via conda (fast, reliable) ───────────────────
RUN conda install -y -c conda-forge \
        python=3.11 \
        rdkit=2024.03 \
    && conda clean -afy

# ── Install pip dependencies ──────────────────────────────────────────────────
COPY requirements_dashboard.txt .
RUN pip install --no-cache-dir -r requirements_dashboard.txt

# ── Copy application source ────────────────────────────────────────────────────
COPY src/ ./src/
COPY dashboard/ ./dashboard/

# ── Streamlit configuration ───────────────────────────────────────────────────
RUN mkdir -p /root/.streamlit
COPY .streamlit/ /root/.streamlit/ 2>/dev/null || true

# ── Expose dashboard port ─────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Start Streamlit ───────────────────────────────────────────────────────────
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
