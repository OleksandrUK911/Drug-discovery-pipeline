#!/usr/bin/env python
"""
scripts/export_reports.py
─────────────────────────
Convert Drug Discovery notebooks to HTML reports using nbconvert.

Usage:
    python scripts/export_reports.py           # convert all 7 notebooks
    python scripts/export_reports.py --nb 01   # convert NB01 only
    python scripts/export_reports.py --nb 06 07  # convert NB06 and NB07

Outputs HTML files to reports/ directory.
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── Notebook registry ─────────────────────────────────────────────────────────
NOTEBOOKS: dict[str, dict] = {
    "01": {
        "path": "01_Toxicity_Prediction/Toxicity_Prediction.ipynb",
        "output": "reports/Toxicity_Prediction.html",
        "title": "NB01 — Toxicity Prediction",
    },
    "02": {
        "path": "02_ADMET_Properties/ADMET_Properties_Prediction.ipynb",
        "output": "reports/ADMET_Properties_Prediction.html",
        "title": "NB02 — ADMET Properties",
    },
    "03": {
        "path": "03_Activity_Classification/Activity_Classification.ipynb",
        "output": "reports/Activity_Classification.html",
        "title": "NB03 — Activity Classification",
    },
    "04": {
        "path": "04_Molecule_Generation/Molecule_Generation_VAE.ipynb",
        "output": "reports/Molecule_Generation_VAE.html",
        "title": "NB04 — Molecule Generation",
    },
    "05": {
        "path": "05_Molecular_Clustering/Molecular_Clustering.ipynb",
        "output": "reports/Molecular_Clustering.html",
        "title": "NB05 — Molecular Clustering",
    },
    "06": {
        "path": "06_Structure_Based_Design/Structure_Based_Design.ipynb",
        "output": "reports/Structure_Based_Design_NB.html",
        "title": "NB06 — Structure-Based Design",
    },
    "07": {
        "path": "07_Drug_Repurposing/Drug_Repurposing.ipynb",
        "output": "reports/Drug_Repurposing_NB.html",
        "title": "NB07 — Drug Repurposing",
    },
}

REPO_ROOT = Path(__file__).parent.parent.resolve()


def convert_notebook(nb_id: str, verbose: bool = True) -> bool:
    """Convert a single notebook to HTML via nbconvert. Returns True on success."""
    info = NOTEBOOKS.get(nb_id)
    if info is None:
        print(f"  ❌ Unknown notebook id '{nb_id}'. Valid: {list(NOTEBOOKS)}")
        return False

    nb_path = REPO_ROOT / info["path"]
    out_html = REPO_ROOT / info["output"]

    if not nb_path.exists():
        print(f"  ⚠️  Notebook not found: {nb_path}")
        return False

    out_html.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "html",
        "--template", "lab",
        "--output", str(out_html.name),
        "--output-dir", str(out_html.parent),
        str(nb_path),
    ]

    if verbose:
        print(f"  📄 {info['title']}")
        print(f"       {nb_path.relative_to(REPO_ROOT)} → {out_html.relative_to(REPO_ROOT)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        size_kb = out_html.stat().st_size // 1024 if out_html.exists() else 0
        if verbose:
            print(f"       ✅ Done ({size_kb} KB)")
        return True
    else:
        print(f"       ❌ nbconvert error:\n{result.stderr[:500]}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Drug Discovery notebooks to HTML reports."
    )
    parser.add_argument(
        "--nb", nargs="*", default=None,
        help="Notebook IDs to convert (e.g. 01 06 07). Omit to convert all.",
    )
    parser.add_argument(
        "--no-lab", action="store_true",
        help="Use default template instead of 'lab' (smaller output).",
    )
    args = parser.parse_args()

    # Check nbconvert available
    chk = subprocess.run(
        [sys.executable, "-m", "nbconvert", "--version"],
        capture_output=True, text=True,
    )
    if chk.returncode != 0:
        print("❌ nbconvert not found.  Run:  pip install nbconvert")
        sys.exit(1)

    ids_to_convert = args.nb if args.nb else list(NOTEBOOKS.keys())

    print(f"\n🔄 Exporting {len(ids_to_convert)} notebook(s) …\n")
    successes, failures = [], []

    for nb_id in ids_to_convert:
        ok = convert_notebook(nb_id.zfill(2), verbose=True)
        (successes if ok else failures).append(nb_id)

    print(f"\n{'─'*50}")
    print(f"✅ Success : {len(successes)}  ({', '.join(successes) or '—'})")
    if failures:
        print(f"❌ Failed  : {len(failures)}  ({', '.join(failures)})")
    print(f"\n📂 Reports saved to: {REPO_ROOT / 'reports'}")
    print(f"🌐 Open  reports/index.html  in your browser\n")


if __name__ == "__main__":
    main()
