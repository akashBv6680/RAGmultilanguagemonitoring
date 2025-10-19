import streamlit as st
import os
import sys
import tempfile
import uuid
import requests
import time
from datetime import datetime
import re
import shutil

# =====================================================================
# FIX 1: SQLITE3 PATCH
# =====================================================================
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass

# --- Core RAG Imports ---
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# --- Google Gemini Imports ---
from google import genai
from google.genai.errors import APIError

# --- LangSmith Imports ---
from langsmith import traceable, tracing_context

# --- Constants and Configuration ---
COLLECTION_NAME = "rag_documents"

# *** CRITICAL FIX: Read API Key from Streamlit Secrets ***
# This is the line that reads the key from the .streamlit/secrets.toml file
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
GEMINI_MODEL_ID = "gemini-2.5-flash-preview-09-2025" 

if not GEMINI_API_KEY:
    # Use st.warning to show the message on the Streamlit page
    st.warning("🚨 GEMINI_API_KEY is not set in Streamlit Secrets. Please set it to proceed.")
    st.stop() 
    
# Dictionary of supported languages and their ISO 639-1 codes for the LLM
LANGUAGE_DICT = {
    # ... (rest of the dictionary is unchanged)
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": "fr", 
    "German": "de", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn", 
    "Japanese": "ja", "Korean": "ko", "Russian": "ru", 
    "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt", 
    "Italian": "it", "Dutch": "nl", "Turkish": "tr"
}

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================
# ... (rest of the session state initialization is unchanged)
if 'selected_language' not in st.session_state: st.session_state['selected_language'] = 'English'
if 'messages' not in st.session_state: st.session_state.messages = []
if 'chat_history' not in st.session_state: st.session_state.chat_history = {}
if 'current_chat_id' not in st.session_state: st.session_state.current_chat_id = None


@st.cache_resource
def initialize_dependencies():
    """
    Initializes and returns the ChromaDB client, SentenceTransformer model,
    and the Google GenAI Client.
    """
    try:
        # 1. Initialize ChromaDB
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        
        # 2. Initialize Sentence Transformer (for embeddings)
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
        # 3. Initialize Google GenAI Client (for LLM)
        # The key is now passed directly to the client from the st.secrets variable
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        
        return db_client, model, gemini_client
    except Exception as e:
        st.error(f"An error occurred during dependency initialization. Error: {e}")
        st.stop()
        
def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

@traceable(run_type="llm")
def call_gemini_api(prompt, max_retries=3):
    """
    Calls the Google Gemini API for text generation.
    """
    gemini_client = st.session_state.gemini_client
    
    retry_delay = 1
    for i in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL_ID,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                )
            )
            return response.text.strip()
            
        except APIError as e:
            st.warning(f"Gemini API error (try {i+1}/{max_retries}). Retrying in {retry_delay} seconds. Error: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2
        except Exception as e:
            st.error(f"An unexpected error occurred during the API call: {e}")
            return f"Error: {e}"
    
    return "Error: Failed to get a response from the model after multiple retries."


def clear_chroma_data():
# ... (all document and document processing functions are unchanged)
    """Clears all data from the ChromaDB collection."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
            st.session_state.db_client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error(f"Error clearing collection: {e}")

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_documents(text_data, chunk_size=500, chunk_overlap=100):
    """Splits a single string of text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text_data)

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

@traceable(run_type="retriever")
def retrieve_documents(query, n_results=5):
    """
    Retrieves the most relevant documents from ChromaDB based on a query.
    """
    collection = get_collection()
    model = st.session_state.model
    
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    if results and results.get('documents') and results['documents'][0]:
        return results['documents'][0]
    return []

@traceable(run_type="chain")
def rag_pipeline(query, selected_language_code):
    """
    Executes the full RAG pipeline with a check for documents.
    """
    collection = get_collection()
    if collection.count() == 0:
        return "Hey there! I'm a chatbot that answers questions based on documents you provide. Please upload a `.txt`, `.pdf` file, or enter a GitHub raw URL in the section above before asking me anything. I'm ready when you are! 😊"

    relevant_docs = retrieve_documents(query)
    
    if not relevant_docs:
        return "I couldn't find relevant information in the uploaded documents to answer your question."

    context = "\n".join(relevant_docs)
    
    prompt = (
        f"You are an expert document assistant. Your task is to answer the 'Question' using ONLY the 'Context' provided below. "
        f"Your final response MUST be in {st.session_state.selected_language}. "
        f"If the Context does not contain the answer, you must politely state that the information is missing. "
        f"\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    )
    
    response = call_gemini_api(prompt)

    if response.startswith("Error:"):
        return response
    
    return response

def display_chat_messages():
# ... (all UI helper functions are unchanged)
    """Displays all chat messages in the Streamlit app."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def is_valid_github_raw_url(url):
    """Checks if the URL is a raw GitHub URL for .txt or .md files."""
    return re.match(r"https://raw\.githubusercontent\.com/.+\.(txt|md)$", url)

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
        
        if st.session_state.current_chat_id and st.session_state.chat_history[st.session_state.current_chat_id]['title'] == "New Chat":
            title = prompt[:50] + ('...' if len(prompt) > 50 else '')
            st.session_state.chat_history[st.session_state.current_chat_id]['title'] = title


# --- Streamlit UI ---
def main_ui():
    """Sets up the main Streamlit UI for the RAG chatbot."""
    st.set_page_config(
        page_title="RAG Chat Flow (Gemini LLM)", 
        layout="wide",
        initial_sidebar_state="auto"
    )

    # Initialize dependencies: db_client, model, and gemini_client
    if 'db_client' not in st.session_state or 'model' not in st.session_state or 'gemini_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client = initialize_dependencies()
        
    if 'current_chat_id' not in st.session_state or not st.session_state.messages:
        new_chat_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_chat_id
        st.session_state.messages = []
        st.session_state.chat_history[new_chat_id] = {
            'messages': st.session_state.messages,
            'title': "New Chat",
            'date': datetime.now()
        }

    # Sidebar
    with st.sidebar:
        st.header("RAG Chat Flow")
        
        st.caption(f"LLM: **{GEMINI_MODEL_ID}**")
        
        st.session_state.selected_language = st.selectbox(
            "Select Response Language",
            options=list(LANGUAGE_DICT.keys()),
            key="language_selector"
        )
        
        if st.button("New Chat", use_container_width=True):
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
                
                is_current = chat_id == st.session_state.current_chat_id
                style = "background-color: #262730; border-radius: 5px; padding: 10px;" if is_current else "padding: 10px;"
                
                st.markdown(
                    f"<div style='{style}'>",
                    unsafe_allow_html=True
                )
                if st.button(f"{chat_title}", key=f"btn_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.messages = st.session_state.chat_history[chat_id]['messages']
                    st.experimental_rerun()
                st.markdown(f"<small>{date_str}</small></div>", unsafe_allow_html=True)

    # Main content area
    st.title("📚 Retrieval Augmented Generation (RAG) Chatbot")
    st.info("Powered by Google Gemini API and monitored via LangSmith.")
    
    # Document upload/processing section
    with st.container():
        st.subheader("Add Context Documents")
        uploaded_files = st.file_uploader(
            "Upload files (.txt, .pdf)", 
            type=["txt", "pdf"], 
            accept_multiple_files=True
        )
        github_url = st.text_input("Enter a GitHub raw `.txt` or `.md` URL:")

        if uploaded_files:
            if st.button(f"Process {len(uploaded_files)} File(s)"):
                with st.spinner("Processing files..."):
                    total_chunks = 0
                    for uploaded_file in uploaded_files:
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        file_contents = None

                        try:
                            if file_ext == "txt":
                                file_contents = uploaded_file.read().decode("utf-8")
                            elif file_ext == "pdf":
                                file_contents = extract_text_from_pdf(uploaded_file)
                            else:
                                st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
                                continue
                            
                            if file_contents:
                                documents = split_documents(file_contents)
                                process_and_store_documents(documents)
                                total_chunks += len(documents)
                                
                        except Exception as e:
                            st.error(f"Failed to process {uploaded_file.name}: {e}")
                            continue

                    if total_chunks > 0:
                        st.success(f"Successfully processed {len(uploaded_files)} file(s) into {total_chunks} chunks!")
                    else:
                        st.warning("No new content was added to the knowledge base.")


        if github_url:
            if st.button("Process URL"):
                if not is_valid_github_raw_url(github_url):
                    st.error("Invalid URL format. Please use a raw GitHub URL ending in `.txt` or `.md`.")
                else:
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
    
    st.markdown("---")
    
    # Chat display and input
    display_chat_messages()
    handle_user_input()

if __name__ == "__main__":
    main_ui()
