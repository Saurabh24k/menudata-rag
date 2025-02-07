# config.py (updated)
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"

PATHS = {
    "data_dir": DATA_DIR,
    "menu_data": DATA_DIR / "menu.csv",
    "external_dir": DATA_DIR / "external",
    "faiss_index": DATA_DIR / "faiss_index",
    "bm25_index": DATA_DIR / "bm25_index.pkl",
    "chunks": DATA_DIR / "chunks.pkl",
    "metadata": DATA_DIR / "metadata.pkl"
}

MODELS = {
    "embedding": "sentence-transformers/all-MiniLM-L6-v2",
    "llm": "mistralai/Mistral-7B-Instruct-v0.3"
}

HF_API_URL = f"https://api-inference.huggingface.co/models/{MODELS['llm']}"

CHUNK_SETTINGS = {
    "max_length": 256,
    "overlap": 32,
    "separators": ["\n\n", "\n", ". ", "! ", "? ", ", "]
}

API_KEYS = {
    "news": os.getenv("NEWS_API_KEY"),
    "hf": os.getenv("HF_TOKEN")
}

REFERENCE_TYPES = {
    "wikipedia": {
        "name": "Wikipedia",
        "url_template": "https://en.wikipedia.org/wiki/{title}"
    },
    "news": {
        "name": "News Article",
        "url_template": "{url}"
    },
    "menu": {
        "name": "Menu Database",
        "url_template": "https://drive.google.com/uc?id=1a9Of5BjLWdhPHPVkdXNgUYzStMLlgwZr"  # internal URL if available
    }
}

CITATION_SETTINGS = {
    "max_sources": 5,  # Maximum references to show
    "min_confidence": 0.3,  # Minimum similarity score to include as reference
    "source_labels": {
        "menu": "Restaurant Menu",
        "news": "News Article",
        "wikipedia": "Wikipedia"
    }
}

# Hybrid Search Weight Settings
HYBRID_SEARCH = {
    "bm25_weight": 0.4,  # Weight for BM25 (Lexical Search)
    "faiss_weight": 0.6  # Weight for FAISS (Semantic Search)
}

# Validate API Keys
if not API_KEYS["news"]:
    print("⚠️ Warning: NEWS_API_KEY is missing! News search might not work.")

if not API_KEYS["hf"]:
    print("⚠️ Warning: HF_TOKEN is missing! LLM API calls might fail.")


DATA_DIR.mkdir(parents=True, exist_ok=True)
PATHS["external_dir"].mkdir(parents=True, exist_ok=True)