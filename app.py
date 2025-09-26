import streamlit as st
import os
import sys
import tempfile
import uuid
import json
import requests
import time
from datetime import datetime
import re
import shutil

# --- LangSmith Imports ---
from langsmith import Client, traceable, tracing_context
# Note: For LangSmith tracing to work, the following environment variables MUST be set:
# LANGCHAIN_TRACING_V2 = "true"
# LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
# LANGCHAIN_API_KEY = "your-langsmith-api-key"
# LANGCHAIN_PROJECT = "your-project-name"


# This block MUST be at the very top to fix the sqlite3 version issue.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    st.error("pysqlite3 is not installed. Please add 'pysqlite3-binary' to your requirements.txt.")
    st.stop()

# Now import chromadb and other libraries
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Constants and Configuration ---
COLLECTION_NAME = "rag_documents"
# API key is provided by the user
# NOTE: It's best practice to load API keys from environment variables or Streamlit secrets
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY") 
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Dictionary of supported languages and their ISO 639-1 codes for the LLM
LANGUAGE_DICT = {
    "English": "en",
    "Spanish": "es",
    "Arabic": "ar",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Tamil": "ta",
    "Bengali": "bn",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Turkish": "tr"
}

@st.cache_resource
def initialize_dependencies():
    """
    Initializes and returns the ChromaDB client and SentenceTransformer model.
    Using @st.cache_resource ensures this runs only once.
    """
    try:
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        # Explicitly load the model to the CPU to avoid PyTorch-related errors
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        return db_client, model
    except Exception as e:
        st.error(f"An error occurred during dependency initialization: {e}.")
        st.stop()
        
def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

# 1. ADD @traceable for LLM call
@traceable(run_type="llm")
def call_together_api(prompt, max_retries=5):
    """
    Calls the Together AI API with exponential backoff for retries.
    
    The @traceable decorator creates an 'llm' run in LangSmith.
    The 'prompt' argument serves as the input to the run.
    """
    retry_delay = 1
    for i in range(max_retries):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOGETHER_API_KEY}"
            }
            # Together AI expects 'messages' format
            messages_payload = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            payload = {
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "messages": messages_payload,
                "temperature": 0.7,
                "max_tokens": 1024
            }
            
            response = requests.post(TOGETHER_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            
            # Extract and return the completion content
            response_json = response.json()
            return response_json['choices'][0]['message']['content'] # LangSmith captures this as the run's output
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                st.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            elif e.response.status_code == 401:
                st.error("Invalid API Key. Please check your Together AI API key.")
                return f"Error: 401 Unauthorized"
            else:
                st.error(f"Failed to call API after {i+1} retries: {e}")
                return f"Error: {e}"
        except Exception as e:
            st.error(f"An error occurred during the API call: {e}")
            return f"Error: {e}"

def clear_chroma_data():
    """Clears all data from the ChromaDB collection."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error(f"Error clearing collection: {e}")

def split_documents(text_data, chunk_size=500, chunk_overlap=100):
    """Splits a single string of text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text_data)

def is_valid_github_raw_url(url):
    """Checks if a URL is a valid GitHub raw file URL."""
    pattern = r"https://raw\.githubusercontent\.com/[\w-]+/[\w-]+/[^/]+/[\w./-]+\.(txt|md)"
    return re.match(pattern, url) is not None

def process_and_store_documents(documents):
    """
    Processes a list of text documents, generates embeddings, and
    stores them in ChromaDB.
    """
    collection = get_collection()
    model = st.session_state.model

    embeddings = model.encode(documents).tolist()
    document_ids = [str(uuid.uuid4()) for _ in documents]
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=document_ids
    )

    st.toast("Documents processed and stored successfully!", icon="✅")

# 2. ADD @traceable for Retriever call
@traceable(run_type="retriever")
def retrieve_documents(query, n_results=5):
    """
    Retrieves the most relevant documents from ChromaDB based on a query.
    The @traceable decorator creates a 'retriever' run in LangSmith.
    """
    collection = get_collection()
    model = st.session_state.model
    
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results['documents'][0]

# 3. ADD @traceable for the entire RAG pipeline (Chain)
@traceable(run_type="chain")
def rag_pipeline(query, selected_language_code):
    """
    Executes the full RAG pipeline with a check for documents.
    The @traceable decorator creates a 'chain' run in LangSmith, which
    will nest the 'retriever' and 'llm' calls.
    """
    collection = get_collection()
    if collection.count() == 0:
        return "Hey there! I'm a chatbot that answers questions based on documents you provide. Please upload a `.txt` file or enter a GitHub raw URL in the section above before asking me anything. I'm ready when you are! 😊"

    # Calls the decorated retrieve_documents function (creates a nested 'retriever' run)
    relevant_docs = retrieve_documents(query)
    
    context = "\n".join(relevant_docs)
    prompt = f"Using the following information, answer the user's question. The final response MUST be in {st.session_state.selected_language}. If the information is not present, state that you cannot answer. \n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Calls the decorated call_together_api function (creates a nested 'llm' run)
    response = call_together_api(prompt)

    if response.startswith("Error:"):
        return "An error occurred while generating the response. Please try again."
    
    return response

def display_chat_messages():
    """Displays all chat messages in the Streamlit app."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    """Handles new user input, runs the RAG pipeline, and updates chat history."""
    if prompt := st.chat_input("Ask about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                selected_language_code = LANGUAGE_DICT[st.session_state.selected_language]
                response = rag_pipeline(prompt, selected_language_code)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Streamlit UI ---
def main_ui():
    """Sets up the main Streamlit UI for the RAG chatbot."""
    st.set_page_config(layout="wide")

    # Sidebar
    with st.sidebar:
        st.header("RAG Chat Flow")
        st.session_state.selected_language = st.selectbox(
            "Select a Language",
            options=list(LANGUAGE_DICT.keys()),
            key="language_selector"
        )
        
        if st.button("New Chat"):
            st.session_state.messages = []
            clear_chroma_data()
            st.session_state.chat_history = {}
            st.session_state.current_chat_id = None
            st.experimental_rerun()

        st.subheader("Chat History")
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            sorted_chat_ids = sorted(
                st.session_state.chat_history.keys(), 
                key=lambda x: st.session_state.chat_history[x]['date'], 
                reverse=True
            )
            for chat_id in sorted_chat_ids:
                chat_title = st.session_state.chat_history[chat_id]['title']
                date_str = st.session_state.chat_history[chat_id]['date'].strftime("%b %d, %I:%M %p")
                if st.button(f"**{chat_title}** - {date_str}", key=chat_id):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.messages = st.session_state.chat_history[chat_id]['messages']
                    st.experimental_rerun()

    # Main content area
    st.title("RAG Chat Flow")
    st.markdown("---")
    
    # Initialize dependencies outside of the main UI block to prevent re-initialization
    if 'db_client' not in st.session_state or 'model' not in st.session_state:
        st.session_state.db_client, st.session_state.model = initialize_dependencies()

    # Document upload/processing section
    with st.container():
        st.subheader("Add Context Documents")
        uploaded_files = st.file_uploader("Upload text files (.txt)", type="txt", accept_multiple_files=True)
        github_url = st.text_input("Enter a GitHub raw `.txt` or `.md` URL:")

        if uploaded_files:
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    for uploaded_file in uploaded_files:
                        file_contents = uploaded_file.read().decode("utf-8")
                        documents = split_documents(file_contents)
                        process_and_store_documents(documents)
                    st.success("All files processed and stored successfully! You can now ask questions about their content.")

        if github_url and is_valid_github_raw_url(github_url):
            if st.button("Process URL"):
                with st.spinner("Fetching and processing file from URL..."):
                    try:
                        response = requests.get(github_url)
                        response.raise_for_status()
                        file_contents = response.text
                        documents = split_documents(file_contents)
                        process_and_store_documents(documents)
                        st.success("File from URL processed! You can now chat about its contents.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error fetching URL: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.session_state.chat_history[st.session_state.current_chat_id] = {
            'messages': st.session_state.messages,
            'title': "New Chat",
            'date': datetime.now()
        }

    display_chat_messages()
    handle_user_input()

if __name__ == "__main__":
    main_ui()
