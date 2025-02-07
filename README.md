# 🍽️ MenuData RAG - AI-Powered Restaurant Assistant

Welcome to **MenuData RAG**, an AI-powered restaurant guide that helps you explore menu items, compare restaurant trends, and get real-time insights on food choices using advanced retrieval-augmented generation (RAG) techniques.

## 🚀 Live Demo
[Click here](#) to try it out! *(https://huggingface.co/spaces/Sau24k/menudata-RAG)*

## ✨ Features
- 🔍 **Search for Menu Items** – Find restaurants serving specific dishes like gluten-free pizza or Impossible Burgers.
- 📈 **Trend Analysis** – Get insights into trending ingredients, flavors, and dining preferences.
- 📚 **AI-Powered Knowledge** – Retrieves data from **restaurant menus, Wikipedia, and news sources** for well-rounded answers.
- ⚡ **Fast & Reliable** – Combines **FAISS (semantic search) and BM25 (lexical search)** for efficient information retrieval.
- 🏆 **Cited Responses** – Provides source citations for transparency and reliability.

## 🛠️ How It Works
1. **User asks a question** (e.g., "Where can I find sushi in San Francisco?").
2. **AI retrieves relevant data** from:
   - Internal **restaurant menu database** 📜
   - **Wikipedia & News sources** 📰
3. **FAISS & BM25 hybrid search** ranks the best results for semantic and keyword-based relevance.
4. **Response is generated** with citations for transparency and accuracy.

## 📦 Tech Stack
- **Framework**: Streamlit
- **LLM**: Mistral-7B
- **Retrieval Mechanisms**:
  - **FAISS** (Semantic search for similar content)
  - **BM25** (Lexical search for keyword matches)
- **Data Sources**:
  - Google Drive (for structured menu datasets)
  - Wikipedia & News APIs (for external knowledge retrieval)
- **Hosting**: Hugging Face Spaces

## 💡 Have feedback or suggestions?
Feel free to reach out at **[saurabhrajput24k@gmail.com](mailto:saurabhrajput24k@gmail.com)**!

