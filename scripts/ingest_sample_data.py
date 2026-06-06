"""
Downloads a sample SEC 10-K filing and ingests it into Docustra.
Run: uv run python scripts/ingest_sample_data.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Apple 10-K 2023 — publicly available from SEC EDGAR
SEC_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
SAMPLE_PDF_PATH = DATA_DIR / "apple_10k_sample.pdf"


def download_sample() -> Path:
    if SAMPLE_PDF_PATH.exists():
        print(f"Sample already exists: {SAMPLE_PDF_PATH}")
        return SAMPLE_PDF_PATH

    print("Note: For demo purposes, create a PDF from any public 10-K filing.")
    print(f"Place it at: {SAMPLE_PDF_PATH}")
    print()
    print("Quick option — download from SEC EDGAR:")
    print("  1. Visit https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AAPL&type=10-K")
    print("  2. Download the latest 10-K as PDF")
    print(f"  3. Save it to: {SAMPLE_PDF_PATH}")
    return SAMPLE_PDF_PATH


def ingest_via_api(file_path: Path) -> None:
    import requests

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"Ingesting: {file_path.name}")
    with open(file_path, "rb") as f:
        response = requests.post(
            "http://localhost:8000/ingest/upload",
            files={"file": (file_path.name, f, "application/pdf")},
            data={"build_graph": "true"},
        )

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Ingestion complete:")
        print(f"   Chunks indexed: {data['chunks_indexed']}")
        print(f"   Images found:   {data['images_found']}")
        print(f"   Graph entities: {data['graph_entities']}")
    else:
        print(f"❌ Ingestion failed: {response.text}")


if __name__ == "__main__":
    path = download_sample()
    if path.exists():
        ingest_via_api(path)
    else:
        print("\nOnce you have a PDF, run this script again to ingest it.")
