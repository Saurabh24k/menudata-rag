import streamlit as st
from query import RestaurantBot
import time
import re

# Import the download-check function
from check_assets import check_and_download_assets

def inject_custom_css():
    st.markdown(
        """
        <style>
            /* Main container */
            .main {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #e6e6e6;
            }
            
            /* Chat messages */
            .user-message {
                background: rgba(255, 255, 255, 0.1) !important;
                border-radius: 15px !important;
                padding: 15px !important;
                margin: 10px 0 !important;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .bot-message {
                background: rgba(29, 78, 137, 0.3) !important;
                border-radius: 15px !important;
                padding: 15px !important;
                margin: 10px 0 !important;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            /* Reference cards */
            .reference-card {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 15px;
                margin-bottom: 10px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .reference-title {
                font-size: 16px;
                font-weight: bold;
                color: #4dabf7;
            }
            .reference-source {
                font-size: 14px;
                color: #ddd;
            }
            .reference-link a {
                color: #ffcc00 !important;
                text-decoration: none !important;
                font-size: 14px;
            }
            .reference-link a:hover {
                text-decoration: underline !important;
            }

            /* Font */
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;600&display=swap');
            * {
                font-family: 'Poppins', sans-serif !important;
            }
        </style>
        """, 
        unsafe_allow_html=True
    )

def extract_references(response_text):
    """Extract references from the response and clean up response text."""
    reference_pattern = r"\[(\d+)\] (.+?): (https?://\S+)"
    matches = re.findall(reference_pattern, response_text)

    references = []
    for match in matches:
        citation_number, title, url = match
        references.append({
            "number": citation_number,
            "title": title,
            "url": url
        })
    
    # Remove references from the response text for better readability
    clean_response = re.sub(reference_pattern, "", response_text).strip()
    
    return clean_response, references

def format_response(response):
    """Formats response with proper line breaks for readability."""
    formatted_response = response.replace(". ", ".\n\n")  # New line after each sentence
    return formatted_response

def main():
    # 1) Ensure big data files are present locally
    check_and_download_assets()
    
    # 2) Inject custom styles
    inject_custom_css()
    
    st.title("🍽️ Culinary Companion")
    st.caption("Your AI-powered Restaurant Guide")
    
    # 3) Initialize the RestaurantBot once
    if 'bot' not in st.session_state:
        with st.spinner('✨ Initializing AI Chef...'):
            st.session_state.bot = RestaurantBot()
    
    # 4) Sidebar info
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/706/706164.png", width=100)
        st.markdown("## Restaurant AI")
        st.markdown("Ask about:")
        st.markdown("- Menu items & ingredients")
        st.markdown("- Restaurant locations")
        st.markdown("- Price ranges & dietary info")
        st.markdown("---")
        st.markdown("**Powered by:**")
        st.markdown("- Mistral-7B LLM")
        st.markdown("- FAISS Vector Search")
        st.markdown("- Real-time News/Wikipedia Data")

        if "references" in st.session_state and st.session_state.references:
            st.markdown("### 📌 References")
            for ref in st.session_state.references:
                st.markdown(f"""
                <div class="reference-card">
                    <div class="reference-title">{ref["title"]}</div>
                    <div class="reference-link"><a href="{ref["url"]}" target="_blank">🔗 View Source</a></div>
                </div>
                """, unsafe_allow_html=True)
    
    # 5) Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 6) Display conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"""
            <div class="{'user-message' if message['role'] == 'user' else 'bot-message'} message-animation">
                {message["content"]}
            </div>
            """, 
            unsafe_allow_html=True)
    
    # 7) Handle user input
    if prompt := st.chat_input("Ask about restaurants..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"""
            <div class="user-message message-animation">
                {prompt}
            </div>
            """, unsafe_allow_html=True)
        
        # Get Bot Response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            with st.spinner(' Analyzing menu data...'):
                response = st.session_state.bot.query(prompt)
            
            # Extract references and clean text
            clean_response, references = extract_references(response)
            formatted_response = format_response(clean_response)
            st.session_state.references = references
            
            # Streaming “typing”
            for chunk in formatted_response.split():
                full_response += chunk + " "
                response_placeholder.markdown(f"""
                <div class="bot-message message-animation">
                    {full_response}▌
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.05)
            
            # Final
            response_placeholder.markdown(f"""
            <div class="bot-message message-animation">
                {formatted_response}
            </div>
            """, unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})

if __name__ == "__main__":
    main()
