# embed.py (fully improved for Hybrid Search)
import numpy as np
import faiss
import torch
import pickle
import logging
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import PATHS, CHUNK_SETTINGS, MODELS
from ingest import load_and_clean_data, get_external_content, build_bm25_index

logger = logging.getLogger(__name__)

def create_embeddings():
    try:
        logger.info("🚀 Starting embedding pipeline...")

        # Check if FAISS index already exists to avoid redundant computation
        if PATHS["faiss_index"].exists():
            logger.info("✅ FAISS index already exists. Skipping embedding computation.")
            return

        # Build BM25 Index (ensures hybrid search works)
        build_bm25_index()

        # Load menu data
        menu_df = load_and_clean_data()
        logger.info(f"📊 Processing {len(menu_df)} valid menu items.")

        # Fetch external content
        external_data = get_external_content("restaurant ingredients")
        if not external_data:
            logger.warning("⚠️ No external content retrieved.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SETTINGS["max_length"],
            chunk_overlap=CHUNK_SETTINGS["overlap"],
            separators=CHUNK_SETTINGS["separators"]
        )

        all_chunks = []
        
        # Process menu items
        for _, row in menu_df.iterrows():
            chunks = text_splitter.split_text(row['full_text'])
            all_chunks.extend([
                (chunk, {
                    "source": "menu",
                    "restaurant": row['restaurant_name'],
                    "item_id": row['item_id'],
                    "menu_item": row.get('menu_item', ''),
                    "category": row.get('menu_category', ''),
                }) for chunk in chunks
            ])

        # Process external content
        for content_type in ["news", "wikipedia"]:
            for content, meta in external_data.get(content_type, []):
                ext_chunks = text_splitter.split_text(content)
                all_chunks.extend([
                    (chunk, {
                        "source": content_type,
                        "title": meta.get("title", ""),
                        "url": meta.get("url", ""),
                    }) for chunk in ext_chunks
                ])

        if not all_chunks:
            logger.warning("⚠️ No valid chunks found. Exiting.")
            return

        texts, metadatas = zip(*all_chunks)

        with open(PATHS["chunks"], "wb") as f:
            pickle.dump(texts, f)
        with open(PATHS["metadata"], "wb") as f:
            pickle.dump(metadatas, f)

        logger.info(f"💾 Saved {len(texts)} chunks and metadata.")

        # Initialize embedding model
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"🖥️ Using device: {device}")

        model = SentenceTransformer(MODELS["embedding"], device=device)
        batch_size = 128
        logger.info(f"🧠 Generating embeddings in batches of {batch_size}...")

        embeddings = model.encode(
            list(texts),
            batch_size=batch_size,
            device=device,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Create and save FAISS index
        logger.info("🔍 Creating FAISS index...")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        faiss.write_index(index, str(PATHS["faiss_index"]))
        logger.info(f"✅ FAISS index saved with {index.ntotal} vectors.")

    except Exception as e:
        logger.error(f"🔥 Embedding pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    create_embeddings()
