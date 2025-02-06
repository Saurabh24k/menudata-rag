# query.py
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from config import PATHS, MODELS, API_KEYS, HF_API_URL, CITATION_SETTINGS, CHUNK_SETTINGS
import pickle
import logging
import re
import time
import torch
import hashlib
from typing import List, Dict, Tuple

from ingest import fetch_wikipedia, fetch_news, load_and_clean_data
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag.log')
    ]
)
logger = logging.getLogger(__name__)

class RestaurantBot:
    def __init__(self):
        self.device = self._get_device()
        self._load_data()
        self._init_models()
        self.headers = {"Authorization": f"Bearer {API_KEYS['hf']}"}
        self.api_timeout = 25
        self._verify_api_access()

        self.source_counter = 1
        self.source_map = {}

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SETTINGS["max_length"],
            chunk_overlap=CHUNK_SETTINGS["overlap"],
            separators=CHUNK_SETTINGS["separators"]
        )

        try:
            self.raw_df = load_and_clean_data()
            logger.info("🔢 Raw data loaded for numeric analysis.")
        except Exception as e:
            logger.error(f"❌ Failed to load raw DataFrame: {str(e)}")
            self.raw_df = None

    def _get_device(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"🖥️ Hardware: {device.upper()}")
        return device

    def _load_data(self):
        try:
            self.index = faiss.read_index(str(PATHS["faiss_index"]))
            with open(PATHS["chunks"], "rb") as f:
                self.all_texts = pickle.load(f)
            with open(PATHS["metadata"], "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"📚 Loaded {len(self.all_texts)} text chunks.")
        except Exception as e:
            logger.error(f"🔴 Data load failed: {str(e)}", exc_info=True)
            raise

    def _init_models(self):
        try:
            self.embed_model = SentenceTransformer(MODELS["embedding"], device=self.device)
            logger.info("🔧 Embedding model initialized.")
        except Exception as e:
            logger.error(f"🔧 Model init failed: {str(e)}", exc_info=True)
            raise

    def _verify_api_access(self):
        try:
            model_id = MODELS['llm']
            status_url = f"https://api-inference.huggingface.co/models/{model_id}"
            response = requests.get(status_url, headers=self.headers, timeout=10)
            status = response.json()
            if not status.get("pipeline_tag"):
                raise RuntimeError(f"Model not ready or not available: {status}")
            logger.info("🔌 Model ready.")
        except Exception as e:
            logger.error(f"🔴 Model check failed: {str(e)}", exc_info=True)
            raise

    # ----------------------------------------------------------------
    #   Public Query Method
    # ----------------------------------------------------------------

    def query(self, question: str, max_retries=3) -> str:
        try:
            start_time = time.time()
            context, relevant_sources = self._get_context(question)
            prompt = self._build_prompt(question, context)

            logger.info(f"PROMPT LENGTH: {len(prompt)} chars")
            response_text = self._get_api_response(prompt, max_retries)
            cleaned_response = self._clean_response(response_text, relevant_sources)

            elapsed = time.time() - start_time
            logger.info(f"⏱️ Query processed in {elapsed:.1f}s")
            return cleaned_response
        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}", exc_info=True)
            return "⚠️ Error processing request"

    # ----------------------------------------------------------------
    #   Real-Time External Fetch
    # ----------------------------------------------------------------

    def _fetch_and_chunk_external(self, user_query: str) -> List[Tuple[np.ndarray, str, dict]]:
        wiki_data = fetch_wikipedia(user_query)
        news_data = fetch_news(user_query)
        combined = [("wikipedia", x[0], x[1]) for x in wiki_data] + \
                   [("news", x[0], x[1]) for x in news_data]

        ephemeral_chunks = []
        for source_type, content, meta in combined:
            chunks = self.text_splitter.split_text(content)
            for chunk in chunks:
                chunk_meta = {
                    "source": source_type,
                    "title": meta.get("title", ""),
                    "url": meta.get("url", "")
                }
                chunk_vector = self.embed_model.encode(
                    [chunk], device=self.device, normalize_embeddings=True
                )
                ephemeral_chunks.append((chunk_vector[0], chunk, chunk_meta))

        return ephemeral_chunks

    def _search_external_chunks(
        self,
        query_embed: np.ndarray,
        external_chunks: List[Tuple[np.ndarray, str, dict]],
        top_k=2
    ) -> List[Tuple[float, str, dict]]:
        scores = []
        for (chunk_embedding, chunk_text, chunk_meta) in external_chunks:
            score = float(np.dot(query_embed, chunk_embedding))
            scores.append((score, chunk_text, chunk_meta))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

    # ----------------------------------------------------------------
    #   Numeric Analysis
    # ----------------------------------------------------------------

    def _maybe_compute_numeric_analysis(self, question: str) -> str:
        """Check if question is about 'average price' etc., then do stats if city column exists."""
        if self.raw_df is None:
            return ""  # no data

        question_lower = question.lower()
        # If not mentioning average/price, skip
        if not ("average price" in question_lower or ("compare" in question_lower and "price" in question_lower)):
            return ""

        # If 'city' column is missing, skip city-based logic
        if 'city' not in self.raw_df.columns:
            logger.warning("⚠️ 'city' column not found, skipping city-based price analysis.")
            # We do a global average across the entire data
            avg_price = self._compute_average_price(self.raw_df)
            return f"The overall average price (no city data) is {avg_price}."

        # Attempt city & category logic
        possible_cities = self.raw_df['city'].dropna().unique().tolist()
        city_found = None
        for c in possible_cities:
            if c.lower() in question_lower:
                city_found = c
                break

        known_categories = ["vegan", "mexican", "italian", "thai", "chinese", "indian", "sushi", "pizza"]
        categories_found = []
        for cat in known_categories:
            if cat in question_lower:
                categories_found.append(cat)

        # Distinguish single vs multiple categories
        if len(categories_found) >= 2:
            cat1, cat2 = categories_found[:2]
            df_cat1 = self.raw_df[self.raw_df['menu_category'].str.lower().str.contains(cat1, na=False)]
            df_cat2 = self.raw_df[self.raw_df['menu_category'].str.lower().str.contains(cat2, na=False)]
            if city_found:
                df_cat1 = df_cat1[df_cat1['city'].str.lower() == city_found.lower()]
                df_cat2 = df_cat2[df_cat2['city'].str.lower() == city_found.lower()]

            avg_price_cat1 = self._compute_average_price(df_cat1)
            avg_price_cat2 = self._compute_average_price(df_cat2)
            return (f"For {city_found if city_found else 'all cities'}, the average price for "
                    f"{cat1.title()} is {avg_price_cat1}, while for {cat2.title()} it is {avg_price_cat2}.")

        elif len(categories_found) == 1:
            cat1 = categories_found[0]
            df_cat1 = self.raw_df[self.raw_df['menu_category'].str.lower().str.contains(cat1, na=False)]
            if city_found:
                df_cat1 = df_cat1[df_cat1['city'].str.lower() == city_found.lower()]
            avg_price_cat1 = self._compute_average_price(df_cat1)
            return (f"For {city_found if city_found else 'all cities'}, the average price for "
                    f"{cat1.title()} is {avg_price_cat1}.")

        # If user asked average price but no category found
        if city_found:
            df_city = self.raw_df[self.raw_df['city'].str.lower() == city_found.lower()]
            avg_price = self._compute_average_price(df_city)
            return f"The average price in {city_found} across all restaurants is {avg_price}."
        else:
            avg_price = self._compute_average_price(self.raw_df)
            return f"The overall average price (all data) is {avg_price}."

    def _compute_average_price(self, df_segment):
        """Try to parse 'price' column as numeric or approximate based on $ signs."""
        if df_segment.empty:
            return "N/A (no data)"

        numeric_prices = []
        for val in df_segment['price']:
            val = val.strip()
            try:
                numeric_val = float(val)
                numeric_prices.append(numeric_val)
                continue
            except:
                pass
            
            if re.match(r'^\${1,4}$', val):
                approx = 10 * len(val)  # e.g. '$$' => 20
                numeric_prices.append(float(approx))

        if not numeric_prices:
            return "N/A (no valid price)"
        
        return f"${round(sum(numeric_prices) / len(numeric_prices), 2)}"

    # ----------------------------------------------------------------
    #   Context Building
    # ----------------------------------------------------------------

    def _get_context(self, query: str, k=5) -> tuple:
        query_embed = self.embed_model.encode([query], device=self.device, normalize_embeddings=True)[0]

        # Numeric analysis
        numeric_text = self._maybe_compute_numeric_analysis(query)
        numeric_chunk = []
        if numeric_text:
            numeric_chunk.append((0.95, numeric_text, {
                "source": "numeric",
                "url": "",
                "title": "Numeric Analysis"
            }))

        # Static FAISS
        scores, indices = self.index.search(query_embed.reshape(1, -1).astype('float32'), k)
        static_results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= len(self.metadata):
                continue
            if score < CITATION_SETTINGS["min_confidence"]:
                continue
            static_results.append((float(score), self.all_texts[idx], self.metadata[idx]))

        # Ephemeral
        ephemeral_chunks = self._fetch_and_chunk_external(query)
        ephemeral_results = self._search_external_chunks(query_embed, ephemeral_chunks, top_k=2)

        # Combine
        combined = numeric_chunk + static_results + ephemeral_results
        combined.sort(key=lambda x: x[0], reverse=True)

        # Optional post-filter
        filtered = self._post_filter_chunks(query, combined)
        if not filtered:
            fallback_text = "No relevant data found for your query."
            filtered = [(0.9, fallback_text, {"source": "fallback"})]

        final = filtered[:k]

        # Build context
        context_lines = []
        relevant_sources = []
        for (score, context_text, meta) in final:
            source_id = self._build_source_id(context_text, meta)
            if source_id not in self.source_map:
                self.source_map[source_id] = self.source_counter
                self.source_counter += 1

            citation_num = self.source_map[source_id]
            context_lines.append(f"[{citation_num}] {context_text}")
            relevant_sources.append((citation_num, meta))

        context_str = "\n".join(context_lines)
        return context_str, relevant_sources

    def _post_filter_chunks(self, query: str, chunk_tuples: List[Tuple[float, str, dict]]) -> List[Tuple[float, str, dict]]:
        """Remove obviously off-topic chunks if no keyword overlap with query."""
        query_keywords = re.findall(r'\w+', query.lower())
        filtered_list = []
        for score, text, meta in chunk_tuples:
            if meta.get('source') == 'fallback':
                filtered_list.append((score, text, meta))
                continue
            text_lower = text.lower()
            matched_words = [kw for kw in query_keywords if kw in text_lower]
            if matched_words:
                filtered_list.append((score, text, meta))
        return filtered_list

    def _build_source_id(self, context_text: str, meta: dict) -> str:
        if meta.get('url'):
            return meta['url']
        elif 'item_id' in meta:
            return f"{meta.get('restaurant', 'unknown')}_{meta['item_id']}"
        text_hash = hashlib.md5(context_text.encode()).hexdigest()
        return f"{meta.get('source', 'unknown')}_{text_hash}"

    # ----------------------------------------------------------------
    #   Prompt & LLM
    # ----------------------------------------------------------------

    def _build_prompt(self, question: str, context: str) -> str:
        instruction = """<s>[INST]
You are a helpful Restaurant Expert Bot. You have been provided with a set of retrieved documents (and nothing else). 
Answer the question ONLY using the context below. If you are not sure of something, respond with "Information not available." 
Never invent references beyond what is provided. Cite sources using [number] notation.

Context:
{context}

Question: {question} [/INST]"""
        return instruction.format(
            context=context if context else "No relevant data found.",
            question=question.strip()
        )

    def _get_api_response(self, prompt: str, max_retries: int) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "return_full_text": False,
                "max_new_tokens": 756,
                "temperature": 0.5,
                "top_p": 0.9
            }
        }
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    HF_API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=self.api_timeout
                )
                if response.status_code == 503:
                    wait_time = min(15 * attempt, 45)
                    logger.info(f"⏳ Model loading (attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("generated_text", "⚠️ Empty response")
                return "⚠️ Unexpected response format"

            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ Attempt {attempt} failed: {str(e)}")
                if attempt == max_retries:
                    return "⚠️ Service unavailable. Please try again later."
                time.sleep(2 ** attempt)
        return "⚠️ Max retries exceeded"

    def _clean_response(self, text: str, relevant_sources: List[tuple]) -> str:
        citations = {}
        for num, meta in relevant_sources:
            source_type = CITATION_SETTINGS["source_labels"].get(meta.get('source', 'menu'), "Menu")
            if meta.get('url'):
                ref = f"[{num}] {source_type}: {meta['url']}"
            elif meta.get('restaurant'):
                ref = f"[{num}] {source_type}: {meta['restaurant']} - {meta.get('menu_item', '')}"
            else:
                ref = f"[{num}] {source_type}"
            citations[num] = ref

        citation_list = [citations[n] for n in sorted(citations.keys())]
        ref_block = "\n".join(citation_list[:CITATION_SETTINGS["max_sources"]])

        cleaned = re.sub(r"<s>|</s>|\[INST\]|\[/INST\]", "", text).strip()
        if citation_list:
            cleaned += f"\n\n**References:**\n{ref_block}"
        return cleaned.strip()

if __name__ == "__main__":
    print("🚀 Starting RestaurantBot in CLI mode...")
    try:
        bot = RestaurantBot()
        print("✅ System ready! Type 'exit' to quit.")
        while True:
            user_q = input("\nUser: ").strip()
            if user_q.lower() in ["exit", "quit"]:
                break
            reply = bot.query(user_q)
            print(f"\nAssistant: {reply}\n")
            print("-"*60)
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
    finally:
        print("\n🔴 Service shutdown.")
