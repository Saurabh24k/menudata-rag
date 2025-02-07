# query.py
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from config import PATHS, MODELS, API_KEYS, HF_API_URL, CITATION_SETTINGS, CHUNK_SETTINGS, HYBRID_SEARCH, REFERENCE_TYPES
import pickle
import logging
import re
import time
import torch
import hashlib
from typing import List, Dict, Tuple
from ingest import fetch_wikipedia, fetch_news, load_and_clean_data, build_bm25_index # Import BM25 build function
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi # Import BM25 for query

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag.log')
    ]
)
logger = logging.getLogger(__name__)

class DataLoadError(Exception):
    """Custom exception for data loading errors."""
    pass

class ModelInitError(Exception):
    """Custom exception for model initialization errors."""
    pass

class APIAccessError(Exception):
    """Custom exception for API access errors."""
    pass


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

        self._init_bm25_index() # Initialize BM25 Index


    def _get_device(self) -> str:
        """Determine and return the device (mps or cpu)."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"🖥️ Hardware: {device.upper()}")
        return device

    def _load_data(self):
        """Load FAISS index, text chunks, and metadata from disk."""
        try:
            self.index = faiss.read_index(str(PATHS["faiss_index"]))
            with open(PATHS["chunks"], "rb") as f:
                self.all_texts = pickle.load(f)
            with open(PATHS["metadata"], "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"📚 Loaded {len(self.all_texts)} text chunks and FAISS index.")
        except Exception as e:
            logger.error(f"🔴 Data load failed: {str(e)}", exc_info=True)
            raise DataLoadError(f"Failed to load data: {e}")

    def _init_models(self):
        """Initialize the SentenceTransformer embedding model."""
        try:
            self.embed_model = SentenceTransformer(MODELS["embedding"], device=self.device)
            logger.info("🔧 Embedding model initialized.")
        except Exception as e:
            logger.error(f"🔧 Embedding model initialization failed: {str(e)}", exc_info=True)
            raise ModelInitError(f"Failed to initialize embedding model: {e}")

    def _verify_api_access(self):
        """Verify access to Hugging Face Inference API."""
        try:
            model_id = MODELS['llm']
            status_url = f"https://api-inference.huggingface.co/models/{model_id}"
            response = requests.get(status_url, headers=self.headers, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            status = response.json()
            if not status.get("pipeline_tag"):
                raise APIAccessError(f"Model not ready or not available: {status}")
            logger.info("🔌 Model ready and API access verified.")
        except requests.exceptions.RequestException as e:
            logger.error(f"🔴 API access check failed: {str(e)}", exc_info=True)
            raise APIAccessError(f"Failed to verify API access: {e}")
        except APIAccessError as e:
            raise e # Re-raise custom APIAccessError
        except Exception as e:
            logger.error(f"🔴 Unexpected error during API access check: {str(e)}", exc_info=True)
            raise APIAccessError(f"Unexpected error verifying API access: {e}")

    def _init_bm25_index(self):
        """Load pre-built BM25 index and documents."""
        try:
            with open(PATHS["bm25_index"], "rb") as f:
                self.bm25_index, self.bm25_documents = pickle.load(f)
            logger.info(f"✅ BM25 index loaded with {len(self.bm25_documents)} documents.")
        except FileNotFoundError:
            logger.warning("⚠️ BM25 index file not found. Building index...")
            build_bm25_index() # Rebuild if missing
            with open(PATHS["bm25_index"], "rb") as f:
                self.bm25_index, self.bm25_documents = pickle.load(f)
            logger.info("✅ BM25 index rebuilt and loaded.")
        except Exception as e:
            logger.error(f"🔴 Failed to load BM25 index: {str(e)}", exc_info=True)
            self.bm25_index = None # Handle gracefully if BM25 fails to load
            self.bm25_documents = [] # Ensure documents are also empty


    # ----------------------------------------------------------------
    #   Public Query Method
    # ----------------------------------------------------------------

    def query(self, question: str, max_retries=3) -> Dict:
        """Main query function orchestrating context retrieval, prompt building, and response generation."""
        try:
            start_time = time.time()
            context, relevant_sources = self._get_context(question)
            prompt = self._build_prompt(question, context)

            logger.info(f"📝 Generated Prompt (length: {len(prompt)} chars):\n{prompt}") # Log full prompt
            response_text = self._get_api_response(prompt, max_retries)
            structured_response = self._clean_response(response_text, relevant_sources)

            elapsed = time.time() - start_time
            logger.info(f"⏱️ Query processed in {elapsed:.1f}s")
            return structured_response
        except APIAccessError as e:
            logger.error(f"API Access Error: {e}")
            return self._build_error_response("Service unavailable due to API issues.")
        except DataLoadError as e:
            logger.error(f"Data Load Error: {e}")
            return self._build_error_response("Data initialization failed. Please check data files.")
        except ModelInitError as e:
            logger.error(f"Model Initialization Error: {e}")
            return self._build_error_response("Model initialization failed. Please check model configuration.")
        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}", exc_info=True)
            return self._build_error_response("An unexpected error occurred during query processing.")


    def _build_error_response(self, message):
        """Build a standard error response."""
        return {
            "response_text": f"⚠️ {message}",
            "sources": []
        }


    # ----------------------------------------------------------------
    #   Real-Time External Fetch
    # ----------------------------------------------------------------

    def _fetch_and_chunk_external(self, query: str) -> List[Tuple[np.ndarray, str, dict]]:
        """Fetch and chunk external content from Wikipedia and News APIs."""
        wiki_data = fetch_wikipedia(query)
        news_data = fetch_news(query)

        # Combine Wikipedia & News results
        combined = [("wikipedia", x[0], x[1]) for x in wiki_data] + [("news", x[0], x[1]) for x in news_data]

        # Process and embed external content
        external_chunks = []
        for source_type, content, meta in combined:
            for chunk in self.text_splitter.split_text(content):
                try:
                    chunk_vector = self.embed_model.encode([chunk], device=self.device, normalize_embeddings=True)[0]
                    external_chunks.append((chunk_vector, chunk, {"source": source_type, "title": meta.get("title", ""), "url": meta.get("url", "#")}))
                except Exception as e:
                    logger.warning(f"⚠️ Failed to embed external chunk from {source_type}: {str(e)}")
                    continue  # Skip failed chunk embeddings

        return external_chunks

    def _search_external_chunks(
        self,
        query_embed: np.ndarray,
        external_chunks: List[Tuple[np.ndarray, str, dict]],
        top_k=2
    ) -> List[Tuple[float, str, dict]]:
        """Search external chunks using cosine similarity."""
        scores = []
        for (chunk_embedding, chunk_text, chunk_meta) in external_chunks:
            try: # Calculate score and handle potential errors numerically
                score = float(np.dot(query_embed, chunk_embedding))
                scores.append((score, chunk_text, chunk_meta))
            except Exception as score_err:
                logger.warning(f"⚠️ Similarity calculation error: {str(score_err)}. Skipping chunk.")
                continue # Skip chunk if scoring fails

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

    # ----------------------------------------------------------------
    #   Numeric Analysis
    # ----------------------------------------------------------------

    def _maybe_compute_numeric_analysis(self, question: str) -> str:
        """Perform numeric analysis if question is price-related and data is available."""
        if self.raw_df is None:
            return ""  # no data

        question_lower = question.lower()
        if not self._is_price_related_question(question_lower):
            return "" # Skip if not price related

        if 'city' not in self.raw_df.columns:
            logger.warning("⚠️ 'city' column missing, defaulting to global price analysis.")
            return self._analyze_global_price() # Global price analysis

        city_found = self._extract_city_from_question(question_lower)
        categories_found = self._extract_categories_from_question(question_lower)

        return self._analyze_price_by_city_category(city_found, categories_found)


    def _is_price_related_question(self, question_lower):
        """Check if the question is related to price analysis."""
        return "average price" in question_lower or ("compare" in question_lower and "price" in question_lower)

    def _extract_city_from_question(self, question_lower):
        """Extract city mentioned in question if available in data."""
        possible_cities = self.raw_df['city'].dropna().unique().tolist()
        for city in possible_cities:
            if city.lower() in question_lower:
                return city
        return None

    def _extract_categories_from_question(self, question_lower):
        """Extract categories mentioned in question."""
        known_categories = ["vegan", "mexican", "italian", "thai", "chinese", "indian", "sushi", "pizza"]
        return [cat for cat in known_categories if cat in question_lower]


    def _analyze_global_price(self):
        """Analyze and return global average price across all data."""
        avg_price = self._compute_average_price(self.raw_df)
        return f"The overall average price (no city-specific data) is {avg_price}."


    def _analyze_price_by_city_category(self, city_found, categories_found):
        """Analyze and return price analysis based on city and categories."""
        if len(categories_found) >= 2:
            return self._compare_category_prices(city_found, categories_found[:2])
        elif len(categories_found) == 1:
            return self._analyze_single_category_price(city_found, categories_found[0])
        elif city_found:
            return self._analyze_city_average_price(city_found)
        else:
            return self._analyze_global_price() # Fallback to global if no specific criteria met


    def _compare_category_prices(self, city_found, categories_found):
        """Compare average prices between two categories, optionally within a city."""
        cat1, cat2 = categories_found
        df_cat1 = self.raw_df[self.raw_df['menu_category'].str.lower().str.contains(cat1, na=False)]
        df_cat2 = self.raw_df[self.raw_df['menu_category'].str.lower().str.contains(cat2, na=False)]
        if city_found:
            df_cat1 = df_cat1[df_cat1['city'].str.lower() == city_found.lower()]
            df_cat2 = df_cat2[df_cat2['city'].str.lower() == city_found.lower()]

        avg_price_cat1 = self._compute_average_price(df_cat1)
        avg_price_cat2 = self._compute_average_price(df_cat2)
        city_str = f" in {city_found}" if city_found else " across all locations"
        return (f"For{city_str}, the average price for {cat1.title()} is {avg_price_cat1}, "
                f"while for {cat2.title()} it is {avg_price_cat2}.")


    def _analyze_single_category_price(self, city_found, category_found):
        """Analyze average price for a single category, optionally within a city."""
        df_cat = self.raw_df[self.raw_df['menu_category'].str.lower().str.contains(category_found, na=False)]
        if city_found:
            df_cat = df_cat[df_cat['city'].str.lower() == city_found.lower()]
        avg_price_cat = self._compute_average_price(df_cat)
        city_str = f" in {city_found}" if city_found else " across all locations"
        return f"For{city_str}, the average price for {category_found.title()} is {avg_price_cat}."

    def _analyze_city_average_price(self, city_found):
        """Analyze average price in a specific city across all restaurants."""
        df_city = self.raw_df[self.raw_df['city'].str.lower() == city_found.lower()]
        avg_price = self._compute_average_price(df_city)
        return f"The average price in {city_found} across all restaurants is {avg_price}."


    def _compute_average_price(self, df_segment):
        """Compute average price from a DataFrame segment, handling various price formats."""
        if df_segment.empty:
            return "N/A (no data)"

        numeric_prices = []
        for val in df_segment['price']:
            val = val.strip()
            try:
                numeric_val = float(val)
                numeric_prices.append(numeric_val)
                continue # Valid numeric price found
            except ValueError:
                pass # Not a direct float

            # Handle $-sign approximations more robustly
            dollar_match = re.match(r'^(\$){1,4}(\.?\d{1,2})?$', val) # e.g., "$$", "$$.99"
            if dollar_match:
                dollar_count = len(dollar_match.group(1)) # Count of $ signs
                fraction_part = dollar_match.group(2) # Fraction part if exists (e.g., ".99")
                approx_price = 10 * dollar_count
                if fraction_part:
                    approx_price += float(fraction_part.lstrip('.')) # Add cents
                numeric_prices.append(float(approx_price))
                continue # Dollar approximation processed


        if not numeric_prices:
            return "N/A (no valid price)" # No prices could be parsed

        return f"${round(sum(numeric_prices) / len(numeric_prices), 2)}"


    # ----------------------------------------------------------------
    #   Context Building - Hybrid Search Integrated
    # ----------------------------------------------------------------

    def _get_context(self, query: str, k=5) -> tuple:
        """Retrieve context using hybrid search: BM25 (Lexical), FAISS (Semantic), and External sources (Wikipedia & News)."""

        query_embed = self.embed_model.encode([query], device=self.device, normalize_embeddings=True)[0]

        # 1️⃣ Numeric Analysis (If applicable)
        numeric_text = self._maybe_compute_numeric_analysis(query)
        numeric_chunk_results = []
        if numeric_text:
            numeric_chunk_results.append((1.2, numeric_text, {"source": "numeric", "title": "Numeric Analysis"}))  # 🔥 Higher priority

        # 2️⃣ BM25 (Lexical) Search
        bm25_results = self._bm25_lexical_search(query, top_k=k)

        # 3️⃣ FAISS (Semantic) Search
        faiss_results = self._faiss_semantic_search(query_embed, top_k=k)

        # 4️⃣ Wikipedia & News (External Content) - **BOOSTED by 50%**
        external_chunks = self._fetch_and_chunk_external(query)
        external_results = self._search_external_chunks(query_embed, external_chunks, top_k=3)
        external_results = [(score * 1.5, text, meta) for score, text, meta in external_results]  # 🔥 Boost relevance by 50%

        # 5️⃣ Hybrid Scoring: Weighted FAISS & BM25
        hybrid_results = self._combine_search_results(
            bm25_results, faiss_results, 
            bm25_weight=HYBRID_SEARCH["bm25_weight"], 
            faiss_weight=HYBRID_SEARCH["faiss_weight"]
        )

        # 6️⃣ Combine Numeric, Hybrid & External Results
        combined_results = numeric_chunk_results + hybrid_results + external_results
        combined_results.sort(key=lambda x: x[0], reverse=True)  # Sort by score

        # 7️⃣ **Force Include At Least 2 Wikipedia/News Entries**
        wiki_news_results = [r for r in external_results if r[2].get("source") in ["wikipedia", "news"]]
        if len(wiki_news_results) < 2:  # Ensure at least 2 Wikipedia or News entries
            additional_needed = 2 - len(wiki_news_results)
            combined_results += wiki_news_results[:additional_needed]

        # 8️⃣ **Post-filtering to Keep Relevant External Chunks**
        filtered_results = self._post_filter_chunks(query, combined_results)
        if not filtered_results:
            filtered_results = [(1.0, "No relevant data found.", {"source": "fallback"})]  # Fallback message

        # 9️⃣ **Final Context Selection (Top `k` results)**
        final_context_chunks = filtered_results[:k]
        context_str, relevant_sources = self._build_context_and_sources(final_context_chunks)

        return context_str, relevant_sources


    def _bm25_lexical_search(self, query: str, top_k: int) -> List[Tuple[float, str, dict]]:
        """Perform BM25 lexical search and return top_k results."""
        if not self.bm25_index: # Safeguard if BM25 index failed to load
            logger.warning("BM25 index not available, skipping BM25 search.")
            return [] # Return empty list if BM25 not available

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query) # Get BM25 scores
        document_score_pairs = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True) # Sort by score

        bm25_results = []
        for doc_id, score in document_score_pairs[:top_k]: # Take top k BM25 results
            if score <= 0: # Skip zero scores
                continue # No relevance
            metadata = self.metadata[doc_id] if doc_id < len(self.metadata) else {"source": "menu"} # Default metadata
            bm25_results.append((float(score), self.bm25_documents[doc_id], metadata)) # Store score, text, metadata

        return bm25_results


    def _faiss_semantic_search(self, query_embed: np.ndarray, top_k: int) -> List[Tuple[float, str, dict]]:
        """Perform FAISS semantic search and return top_k results."""
        scores, indices = self.index.search(query_embed.reshape(1, -1).astype('float32'), top_k)
        faiss_results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= len(self.metadata): # Index out of bounds check
                continue # Skip if index is invalid
            if score < CITATION_SETTINGS["min_confidence"]: # Apply min confidence threshold
                continue # Skip low confidence results

            faiss_results.append((float(score), self.all_texts[idx], self.metadata[idx])) # Store score, text, metadata

        return faiss_results


    def _combine_search_results(self, bm25_results, faiss_results, bm25_weight, faiss_weight):
        """Combine BM25 and FAISS results using weighted scores."""
        combined_results_map = {} # Use a map to avoid duplicates and sum scores

        # Add BM25 results with weight
        for score, text, metadata in bm25_results:
            source_id = self._build_source_id(text, metadata)
            combined_results_map[source_id] = combined_results_map.get(source_id, {"text": text, "metadata": metadata, "score": 0})
            combined_results_map[source_id]["score"] += score * bm25_weight


        # Add FAISS results with weight, summing scores if document already exists
        for score, text, metadata in faiss_results:
            source_id = self._build_source_id(text, metadata)
            combined_results_map[source_id] = combined_results_map.get(source_id, {"text": text, "metadata": metadata, "score": 0})
            combined_results_map[source_id]["score"] += score * faiss_weight


        # Convert map to list of tuples and sort by combined score
        combined_results = [(data["score"], data["text"], data["metadata"]) for data in combined_results_map.values()]
        combined_results.sort(key=lambda x: x[0], reverse=True)
        return combined_results


    def _build_context_and_sources(self, results: List[Tuple[float, str, dict]]) -> Tuple[str, List[Tuple[int, dict]]]:
        """Format retrieved results into readable context with citations."""
        context_lines, sources = [], []
        for idx, (_, text, meta) in enumerate(results):
            citation_num = idx + 1  # Ensure sources are correctly numbered
            context_lines.append(f"[{citation_num}] {text}")
            sources.append((citation_num, meta))  # ✅ Now correctly returns (num, meta) tuples

        return "\n".join(context_lines), sources  # ✅ Correct return type

    


    def _post_filter_chunks(self, query: str, chunk_tuples: List[Tuple[float, str, dict]]) -> List[Tuple[float, str, dict]]:
        """Ensure relevant Wikipedia & news sources are included while filtering irrelevant menu data."""

        query_keywords = set(re.findall(r'\w+', query.lower()))

        filtered_list = []
        wiki_news_count = 0  # Track Wikipedia & news sources

        for score, text, meta in chunk_tuples:
            if meta.get('source') == 'fallback':  # Always keep fallback
                filtered_list.append((score, text, meta))
                continue

            text_keywords = set(re.findall(r'\w+', text.lower()))
            if query_keywords.intersection(text_keywords):  # Keep chunks with keyword overlap
                filtered_list.append((score, text, meta))

            # **Ensure at least 2 Wikipedia or news sources survive filtering**
            if meta.get("source") in ["wikipedia", "news"] and wiki_news_count < 2:
                filtered_list.append((score, text, meta))
                wiki_news_count += 1  # Track count

        return filtered_list



    def _build_source_id(self, context_text: str, meta: dict) -> str:
        """Generate a unique source ID based on metadata and content hash."""
        if meta.get('url'):
            return meta['url'] # URL-based ID for web sources
        elif 'item_id' in meta:
            return f"{meta.get('restaurant', 'unknown')}_{meta['item_id']}" # Menu item ID
        text_hash = hashlib.md5(context_text.encode()).hexdigest() # Hash for generic text content
        return f"{meta.get('source', 'unknown')}_{text_hash}" # Source type + hash


    # ----------------------------------------------------------------
    #   Prompt & LLM Interaction (No significant change here)
    # ----------------------------------------------------------------

    def _build_prompt(self, question: str, context: str) -> str:
        """Construct a prompt for the LLM with explicit priority on Wikipedia & news sources."""

        instruction = """<s>[INST]
    You are a knowledgeable Restaurant Expert and Food Trends Analyst. 
    You have been provided with information from different sources, including restaurant menus, Wikipedia, and news reports.

    ### **Prioritization of Sources**
    - **Wikipedia & News:** These sources contain up-to-date food trends and general food knowledge.
    - **Restaurant Menus:** These sources contain specific menu items but may not provide broader knowledge.

    ### **How to Answer**
    - **If Wikipedia or News provide relevant information, use those sources first.**
    - **If restaurant menus provide an answer, mention that it is from a menu and specify the restaurant.**
    - **If no direct answer is available, try to infer based on related data.**
    - **If nothing is relevant at all, say: "This information is unavailable from the given sources."**

    ---
    ### **Retrieved Context**
    {context}

    ---
    ### **User Question**
    {question}
    [/INST]"""

        return instruction.format(
            context=context if context else "No relevant data found.",
            question=question.strip()
        )



    def _get_api_response(self, prompt: str, max_retries: int) -> str:
        """Get response from Hugging Face Inference API with retry mechanism."""
        payload = {
            "inputs": prompt,
            "parameters": {
                "return_full_text": False,
                "max_new_tokens": 756,
                "temperature": 0.75,
                "top_p": 0.95
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
                response.raise_for_status() # Raise HTTPError for bad responses
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    llm_response = data[0].get("generated_text", "⚠️ Empty response from model")
                    if llm_response.strip() == "": # Check for truly empty responses and handle
                        return "Information not available." # Treat as no info
                    return llm_response # Return valid response

                return "⚠️ Unexpected response format from model" # Handle unexpected API response format

            except requests.exceptions.RequestException as e: # Catch specific request exceptions
                logger.warning(f"⚠️ API Request attempt {attempt}/{max_retries} failed: {str(e)}")
                if response is not None and response.status_code == 503: # Check for 503 status specifically for model loading
                    wait_time = min(15 * attempt, 45)
                    logger.info(f"⏳ Model loading (attempt {attempt}/{max_retries}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt == max_retries: # If max retries reached without success
                    return "⚠️ Service unavailable after multiple retries. Please try again later."
                else: # For other request exceptions, backoff
                    time.sleep(2 ** attempt) # Exponential backoff


        return "⚠️ Max retries exceeded without API response" # Max retries reached without any successful response


    def _clean_response(self, text: str, relevant_sources: List[tuple]) -> Dict:
        """Clean the LLM response and properly format citations."""

        # Format sources with appropriate labels
        citations = []
        for num, meta in relevant_sources:
            citation_entry = {
                "source_type": CITATION_SETTINGS["source_labels"].get(meta.get("source", "menu"), "Menu"),
                "title": meta.get("title", "Unknown Source"),
                "url": meta.get("url", "")
            }

            # If source has a reference URL template, format it
            if meta.get("source") in REFERENCE_TYPES:
                url_template = REFERENCE_TYPES[meta["source"]].get("url_template")
                if url_template:
                    citation_entry["reference_url"] = url_template.format(**meta)

            citations.append(citation_entry)

        # Sort and limit sources
        sorted_sources = citations[:CITATION_SETTINGS["max_sources"]]

        # Clean text output (remove special tokens)
        cleaned_text = re.sub(r"<s>|</s>|\[INST\]|\[/INST\]", "", text).strip()

        return {
            "response_text": cleaned_text,
            "sources": sorted_sources
        }



if __name__ == "__main__":
    print("🚀 Starting RestaurantBot in CLI mode...")
    try:
        bot = RestaurantBot()
        print("✅ System ready! Type 'exit' to quit.")
        while True:
            user_q = input("\nUser: ").strip()
            if user_q.lower() in ["exit", "quit"]:
                break
            reply_json = bot.query(user_q) # Get JSON response
            print(f"\nAssistant: {reply_json['response_text']}\n") # Print response text
            if reply_json['sources']: # Print sources if available
                print("Sources:")
                for source in reply_json['sources'][:CITATION_SETTINGS['max_sources']]: # Limit sources printed in CLI
                    source_str = f"- {source['source_type']}: {source['title']}"
                    if source.get('reference_url'): # Print reference URL if available
                        source_str += f" ({source['reference_url']})"
                    print(source_str)

            print("-"*60)
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user.")
    except (DataLoadError, ModelInitError, APIAccessError) as e: # Catch custom exceptions more specifically
        print(f"\n❌ Startup Error: {str(e)}")
    except Exception as e:
        logger.error(f"🔥 Fatal error during bot operation: {str(e)}", exc_info=True) # Log fatal errors
        print(f"\n❌ Fatal error: {str(e)}") # Inform user about fatal error
    finally:
        print("\n🔴 Service shutdown.")