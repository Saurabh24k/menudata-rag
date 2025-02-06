# ingest.py
import pandas as pd
import requests
from bs4 import BeautifulSoup
from config import PATHS, API_KEYS, REFERENCE_TYPES
from typing import Dict, List, Tuple
import logging
import concurrent.futures
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_str_convert(value) -> str:
    if pd.isna(value):
        return ""
    try:
        return str(value).strip()
    except Exception:
        return ""

def load_and_clean_data() -> pd.DataFrame:
    logger.info("🚀 Loading and cleaning menu data...")
    
    if not PATHS["menu_data"].exists():
        raise FileNotFoundError(f"❌ Menu data missing at {PATHS['menu_data']}")
    
    try:
        df = pd.read_csv(PATHS["menu_data"], engine='python', on_bad_lines='warn')
    except Exception as e:
        logger.error(f"📄 CSV read failed: {str(e)}")
        raise

    # 1) Strip/normalize column names: e.g. " City " -> "city"
    original_cols = list(df.columns)
    df.columns = [col.strip().lower() for col in df.columns]
    logger.info(f"📋 Original columns: {original_cols}")
    logger.info(f"🔎 Normalized columns: {list(df.columns)}")

    # 2) If some columns are missing, create them so code won't break:
    required_cols = [
        'restaurant_name', 'menu_category', 'item_id', 'menu_item',
        'menu_description', 'ingredient_name', 'confidence',
        'address1', 'city', 'zip_code', 'country', 'state',
        'rating', 'review_count', 'price'
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""  # creating empty col if missing
            logger.warning(f"⚠️ Column '{c}' not found, creating empty.")

    # 3) Convert text columns safely
    text_columns = [
        'restaurant_name', 'menu_category', 'menu_item',
        'menu_description', 'ingredient_name',
        'address1', 'city', 'zip_code', 'country', 'state',
        'rating', 'review_count', 'price'
    ]
    for col in text_columns:
        df[col] = df[col].apply(safe_str_convert)

    # 4) Filter by confidence, etc.
    if not pd.api.types.is_numeric_dtype(df['confidence']):
        # convert
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    df = df.dropna(subset=['confidence'])
    df = df[
        (df['confidence'] > 0.65) &
        (df['ingredient_name'] != "") &
        (df['menu_item'] != "")
    ].copy()

    # 5) Build 'full_text'
    def build_full_text(row):
        desc = row['menu_description'] if row['menu_description'] else 'No description available'
        return (
            f"RESTAURANT: {row['restaurant_name']}\n"
            f"CATEGORY: {row['menu_category']}\n"
            f"ITEM: {row['menu_item']}\n"
            f"DESCRIPTION: {desc}\n"
            f"INGREDIENTS: {row['ingredient_name']}\n"
            f"LOCATION: {row['address1']}, {row['city']}, {row['state']} {row['zip_code']}, {row['country']}\n"
            f"RATING: {row['rating']} based on {row['review_count']} reviews, Price: {row['price']}"
        )

    df['full_text'] = df.apply(build_full_text, axis=1)
    logger.info(f"🧹 Cleaned dataset contains {len(df)} entries")

    # 6) Return only columns needed
    final_cols = []
    for c in ['full_text','restaurant_name','item_id','menu_category','menu_item','city','price','rating']:
        if c in df.columns:
            final_cols.append(c)

    return df[final_cols]

def fetch_wikipedia(query: str) -> List[Tuple[str, Dict]]:
    """Fetch Wikipedia content safely."""
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
        response = requests.get(search_url, timeout=15)
        response.raise_for_status()

        results = response.json().get('query', {}).get('search', [])[:2]
        contents = []
        for result in results:
            page_title = result['title'].replace(' ', '_')
            page_url = f"https://en.wikipedia.org/wiki/{page_title}"
            page_response = requests.get(page_url, timeout=15)
            soup = BeautifulSoup(page_response.text, 'html.parser')

            paragraphs = soup.find_all('p')[:3]
            if not paragraphs:
                continue

            content_text = "\n".join([
                re.sub(r'\[\d+\]', '', p.get_text()).strip()
                for p in paragraphs if p.get_text().strip()
            ])
            if content_text:
                contents.append((content_text, {"title": result['title'], "url": page_url}))
        return contents

    except Exception as e:
        logger.error(f"🌐 Wikipedia fetch failed: {str(e)}")
        return []

def fetch_news(query: str) -> List[Tuple[str, Dict]]:
    """Fetch news articles using the NewsAPI."""
    if not API_KEYS["news"]:
        logger.warning("🔑 News API key missing, skipping news fetch.")
        return []

    try:
        news_url = f"https://newsapi.org/v2/everything?q={query}&apiKey={API_KEYS['news']}"
        response = requests.get(news_url, timeout=15)
        response.raise_for_status()

        articles = []
        for article in response.json().get('articles', [])[:3]:
            desc = article.get('description')
            if desc:
                text_block = f"{article['title']}: {desc}"
                meta = {
                    "title": article['title'],
                    "url": article.get('url', ''),
                    "source_name": article.get('source', {}).get('name', 'Unknown')
                }
                articles.append((text_block, meta))
        return articles

    except Exception as e:
        logger.error(f"📰 News fetch failed: {str(e)}")
        return []

def get_external_content(query: str) -> Dict[str, List[Tuple[str, Dict]]]:
    """Fetch wiki & news concurrently."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        wiki_future = executor.submit(fetch_wikipedia, query)
        news_future = executor.submit(fetch_news, query)
        try:
            return {
                "wikipedia": wiki_future.result(timeout=20),
                "news": news_future.result(timeout=20)
            }
        except concurrent.futures.TimeoutError:
            logger.warning("⏰ External data fetch timed out.")
            return {"wikipedia": [], "news": []}
