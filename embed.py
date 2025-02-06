# embed.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import PATHS, CHUNK_SETTINGS, MODELS, REFERENCE_TYPES
from ingest import load_and_clean_data, get_external_content
import torch
import pickle
import logging

logger = logging.getLogger(__name__)

def create_embeddings():
    try:
        logger.info("🚀 Starting embedding pipeline...")

        menu_df = load_and_clean_data()
        logger.info(f"📊 Processing {len(menu_df)} valid menu items")

        logger.info("🌐 Fetching external content (example: 'restaurant ingredients')...")
        external_data = get_external_content("restaurant ingredients")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SETTINGS["max_length"],
            chunk_overlap=CHUNK_SETTINGS["overlap"],
            separators=CHUNK_SETTINGS["separators"]
        )

        all_chunks = []
        # menu items
        for _, row in menu_df.iterrows():
            chunks = text_splitter.split_text(row['full_text'])
            all_chunks.extend([
                (chunk, {
                    "source": "menu",
                    "restaurant": row['restaurant_name'],
                    "item_id": row['item_id'],
                    "menu_item": row.get('menu_item', ''),
                    "category": row.get('menu_category', ''),
                    "reference_type": REFERENCE_TYPES["menu"]
                }) for chunk in chunks
            ])

        # external content
        for content_type in ["news", "wikipedia"]:
            for content, meta in external_data[content_type]:
                ext_chunks = text_splitter.split_text(content)
                all_chunks.extend([
                    (chunk, {
                        "source": content_type,
                        "type": "external",
                        "title": meta.get("title", ""),
                        "url": meta.get("url", ""),
                        "reference_type": REFERENCE_TYPES[content_type]
                    }) for chunk in ext_chunks
                ])

        logger.info(f"✂️ Total chunks generated: {len(all_chunks)}")

        texts, metadatas = zip(*all_chunks) if all_chunks else ([], [])

        with open(PATHS["chunks"], "wb") as f:
            pickle.dump(texts, f)
        with open(PATHS["metadata"], "wb") as f:
            pickle.dump(metadatas, f)
        logger.info("💾 Saved chunks and metadata")

        if len(texts) == 0:
            logger.warning("⚠️ No texts to embed. The dataset might be empty.")
            return

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"🖥️ Using device: {device}")

        model = SentenceTransformer(MODELS["embedding"], device=device)
        batch_size = 64 if device == "mps" else 128
        logger.info(f"🧠 Generating embeddings in batches of {batch_size}...")

        embeddings = model.encode(
            list(texts),
            batch_size=batch_size,
            device=device,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        logger.info("🔍 Creating FAISS index...")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        faiss.write_index(index, str(PATHS["faiss_index"]))
        logger.info(f"✅ Index saved with {index.ntotal} vectors")

    except Exception as e:
        logger.error(f"🔥 Embedding pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    create_embeddings()
