# ═══════════════════════════════════════════════════════════════════
# Drug Discovery Portfolio — Makefile
# ═══════════════════════════════════════════════════════════════════
# Usage:
#   make reports        — convert all 7 notebooks to HTML (reports/)
#   make report NB=01   — convert single notebook (NB=01..07)
#   make dashboard      — run Streamlit dashboard
#   make api            — run FastAPI server
#   make test           — run pytest suite
#   make lint           — run pre-commit hooks
#   make clean          — remove __pycache__ and .pyc files
#   make help           — this message
# ═══════════════════════════════════════════════════════════════════

.PHONY: reports dashboard api test lint clean help

PYTHON      ?= python
JUPYTER     ?= jupyter
STREAMLIT   ?= streamlit
UVICORN     ?= uvicorn
REPORTS_DIR := reports

# ── All notebooks ────────────────────────────────────────────────────────────
NOTEBOOKS := \
	01_Toxicity_Prediction/Toxicity_Prediction.ipynb \
	02_ADMET_Properties/ADMET_Properties_Prediction.ipynb \
	03_Activity_Classification/Activity_Classification.ipynb \
	04_Molecule_Generation/Molecule_Generation_VAE.ipynb \
	05_Molecular_Clustering/Molecular_Clustering.ipynb \
	06_Structure_Based_Design/Structure_Based_Design.ipynb \
	07_Drug_Repurposing/Drug_Repurposing.ipynb

help:
	@echo ""
	@echo "Drug Discovery Portfolio — available targets:"
	@echo ""
	@echo "  make reports       Convert all 7 notebooks to HTML in reports/"
	@echo "  make report NB=01  Convert single notebook (NB=01..07)"
	@echo "  make dashboard     Start Streamlit dashboard  (http://localhost:8501)"
	@echo "  make api           Start FastAPI server        (http://localhost:8000)"
	@echo "  make test          Run pytest test suite"
	@echo "  make lint          Run pre-commit hooks on all files"
	@echo "  make clean         Remove __pycache__ and temporary files"
	@echo ""

# ── HTML reports from all notebooks ─────────────────────────────────────────
reports:
	@echo "📄 Exporting all notebooks to HTML..."
	@$(PYTHON) scripts/export_reports.py
	@echo "✅ HTML reports saved to $(REPORTS_DIR)/"

# ── Single notebook (usage: make report NB=01) ───────────────────────────────
report:
ifndef NB
	$(error NB not set. Usage: make report NB=01)
endif
	@$(PYTHON) scripts/export_reports.py --nb $(NB)

# ── Streamlit dashboard ──────────────────────────────────────────────────────
dashboard:
	$(STREAMLIT) run dashboard/app.py --server.port 8501

# ── FastAPI server ────────────────────────────────────────────────────────────
api:
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── Lint ──────────────────────────────────────────────────────────────────────
lint:
	pre-commit run --all-files

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned"
