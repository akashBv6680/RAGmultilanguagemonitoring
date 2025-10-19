import streamlit as st
import os
import sys
import tempfile
import uuid
import json
import requests
import time
from datetime import datetime
import shutil

# --- Dependency Check and Fix for pysqlite3 ---
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

# --- Local LLM & LangChain Imports ---
# Use the correct LangChain integration for Ollama
from langchain_community.chat_models import ChatOllama 
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

# --- LangSmith Imports (Optional) ---
from langsmith import traceable, tracing_context

# --- Constants and Configuration ---
COLLECTION_NAME = "rag_documents"

# Load configuration from environment variables
# Set this variable before running Streamlit: e.g., export OLLAMA_URL="http://localhost:11434"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434") 
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral") # Ensure this model is pulled in Ollama
LLM_TIMEOUT = 120 # Timeout for the LLM call

# Dictionary of supported languages
LANGUAGE_DICT = {
    "English": "en",
    "Spanish": "es",
    "Arabic": "ar",
    "French": "fr",
    "German": "de",
    # ... (rest of languages omitted for brevity)
    "Turkish": "tr"
}

def is_valid_github_raw_url(url):
    return url.startswith("https://raw.githubusercontent.com/")

@st.cache_resource
def initialize_dependencies():
    """
    Initializes and returns the ChromaDB client, SentenceTransformer model, and Ollama client.
    """
    try:
        # 1. Initialize ChromaDB
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)

        # 2. Initialize Sentence Transformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

        # 3. Initialize Ollama Chat Model
        # CRITICAL: LangChain's ChatOllama connects to the base URL
        ollama_client = ChatOllama(
            base_url=OLLAMA_URL, 
            model=OLLAMA_MODEL,
            temperature=0.7,
            request_timeout=LLM_TIMEOUT,
        )
        
        # Simple test call to verify connection
        ollama_client.invoke([SystemMessage(content="Test message")])
        
        return db_client, model, ollama_client
    
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection Failed: Could not connect to Ollama at {OLLAMA_URL}. Ensure Ollama is running and accessible."
        st.error(f"FATAL ERROR: {error_msg}. Details: {e}")
        st.stop()
    except Exception as e:
        error_msg = f"An error occurred during dependency initialization. Check your Ollama URL/Model. Error: {e}"
        st.error(f"FATAL ERROR: {error_msg}")
        st.stop()

def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

@traceable(run_type="llm")
def call_local_llm(prompt, max_retries=3):
    """
    Calls the local Ollama LLM via LangChain.
    """
    ollama_client = st.session_state.ollama_client

    system_message = SystemMessage(content="You are an expert document assistant. Be concise and accurate.")
    user_message = HumanMessage(content=prompt)

    for i in range(max_retries):
        try:
            with st.spinner(f"Contacting {OLLAMA_MODEL} at {OLLAMA_URL}..."):
                response = ollama_client.invoke([system_message, user_message])
            return response.content

        except requests.exceptions.ConnectionError as e:
            st.warning(f"Connection error to Ollama. Retrying in 2 seconds... (Attempt {i+1}/{max_retries})")
            time.sleep(2)
        except Exception as e:
            st.error(f"An unexpected error occurred during the local LLM call: {e}")
            return f"Error: An unexpected error occurred: {e}"

    st.error("Failed to get a response from local LLM after multiple retries. Check your Ollama service.")
    return "Error: Failed to get response from local LLM."

# --- Document Processing Functions (Kept from previous version) ---

def clear_chroma_data():
    """Clears all data from the ChromaDB collection."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
            st.session_state.db_client.get_or_create_collection(name=COLLECTION_NAME)
            st.toast("Document context cleared!", icon="🧹")
            return True
        return False
    except Exception as e:
        st.error(f"Error clearing collection: {e}")
        return False

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

    with st.spinner("Generating embeddings and storing data..."):
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
def rag_pipeline(query, selected_language):
    """
    Executes the full RAG pipeline with a check for documents.
    """
    collection = get_collection()
    if collection.count() == 0:
        return "Hey there! I'm a chatbot that answers questions based on documents you provide. Please upload a `.txt`, `.pdf` file, or paste text in the section above before asking me anything. I'm ready when you are! 😊"

    relevant_docs = retrieve_documents(query)

    if not relevant_docs:
        return "I couldn't find relevant information in the uploaded documents to answer your question."

    context = "\n".join(relevant_docs)

    prompt = (
        f"You are an expert document assistant. Using ONLY the 'Context' provided below, "
        f"answer the 'Question'. The final response MUST be in {selected_language}. "
        f"If the Context does not contain the answer, politely state that the information is missing. "
        f"\n\nContext:\n---\n{context}\n---\n\nQuestion: {query}\n\nAnswer:"
    )

    response = call_local_llm(prompt)

    if response.startswith("Error:"):
        return response

    return response

# --- Streamlit UI Logic (Simplified) ---

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
            selected_language = st.session_state.selected_language
            response = rag_pipeline(prompt, selected_language)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # Logic to rename the chat title
        if st.session_state.current_chat_id and st.session_state.chat_history[st.session_state.current_chat_id]['title'] == "New Chat":
            title = prompt[:50] + ('...' if len(prompt) > 50 else '')
            st.session_state.chat_history[st.session_state.current_chat_id]['title'] = title

def document_upload_section():
    """UI for document uploading and text input."""
    st.markdown("### 1. Upload Context Documents")
    
    uploaded_file = st.file_uploader(
        "Upload PDF or TXT File",
        type=["pdf", "txt"],
        help="Upload files to use as context for the RAG chatbot."
    )

    text_input = st.text_area(
        "Or Paste Text Directly",
        placeholder="Paste text here to use as context...",
        height=150
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        process_button = st.button("Process Documents & Start Chat", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear All Context", use_container_width=True, help="Clears all uploaded data from memory."):
            clear_chroma_data()
            st.session_state.messages = []
            st.session_state.chat_history = {}
            st.session_state.current_chat_id = None
            st.experimental_rerun()


    if process_button:
        raw_text = ""
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                raw_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                raw_text = uploaded_file.read().decode("utf-8")
        
        if text_input:
            raw_text += "\n" + text_input
            
        if raw_text.strip():
            with st.spinner("Processing document..."):
                documents = split_documents(raw_text)
                process_and_store_documents(documents)
            
            # Start a new chat immediately after processing
            new_chat_id = str(uuid.uuid4())
            st.session_state.current_chat_id = new_chat_id
            st.session_state.messages = [{"role": "assistant", "content": f"I have successfully processed {len(documents)} chunks from your document(s). Ask me a question about the content in {st.session_state.selected_language}!"}]
            st.session_state.chat_history[new_chat_id] = {
                'messages': st.session_state.messages,
                'title': "New Chat",
                'date': datetime.now()
            }
            st.experimental_rerun()
        else:
            st.warning("Please upload a file or paste text before processing.")

def main_ui():
    """Sets up the main Streamlit UI for the RAG chatbot."""
    st.set_page_config(
        page_title="RAG Chat Flow (Local Ollama)",
        layout="wide",
        initial_sidebar_state="auto"
    )

    # Initialize dependencies
    # This call will FAIL if Ollama is not accessible at OLLAMA_URL
    if 'db_client' not in st.session_state or 'model' not in st.session_state or 'ollama_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.ollama_client = initialize_dependencies()

    # Initialize ALL session state variables
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'chat_history' not in st.session_state: st.session_state.chat_history = {}
    if 'selected_language' not in st.session_state: st.session_state.selected_language = "English"

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
        st.header("RAG Chat Flow Configuration")
        st.error("⚠️ Running Local LLM! This will only work if Ollama is running and accessible at the specified URL.")
        st.info(f"LLM: **{OLLAMA_MODEL}** at **{OLLAMA_URL}**")

        st.session_state.selected_language = st.selectbox(
            "Select Response Language",
            options=list(LANGUAGE_DICT.keys()),
            key="language_selector",
            index=list(LANGUAGE_DICT.keys()).index(st.session_state.selected_language)
        )
        
        try:
            doc_count = get_collection().count()
            st.markdown(f"**Loaded Documents:** {doc_count} chunks")
        except:
             st.markdown("**Loaded Documents:** 0 chunks (No context loaded)")

        st.markdown("---")
        st.caption("Chat history is session-based.")
        
    # Main content area
    st.title("📚 Retrieval Augmented Generation (RAG) Chatbot - Local LLM")
    st.subheader("Connects to Local Ollama Service.")
    
    document_upload_section()
    
    st.markdown("---")
    st.subheader("2. Ask Your Question")

    # Chat display and input
    display_chat_messages()
    handle_user_input()

if __name__ == "__main__":
    main_ui()
