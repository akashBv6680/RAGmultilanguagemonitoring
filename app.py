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

# --- Dependency Check and Fix for pysqlite3 ---
try:
    # This block MUST be at the very top to fix the sqlite3 version issue for ChromaDB
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    # If the environment is set up correctly, the normal sqlite3 module will be used.
    pass

# --- Core RAG Imports ---
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# --- Local LLM & LangChain Imports ---
# <--- FIX APPLIED HERE: Changed import from langchain.chat_models to langchain_ollama
from langchain_ollama import ChatOllama 
from langchain.schema import HumanMessage, SystemMessage
from pydantic import ValidationError # Added for better error handling with LangChain

# --- LangSmith Imports ---
from langsmith import Client, traceable, tracing_context

# --- Constants and Configuration ---
COLLECTION_NAME = "rag_documents"

# --- Ollama/Local LLM Configuration ---
OLLAMA_MODEL = "mistral" # Ensure you have pulled this model with 'ollama pull mistral'
OLLAMA_URL = "http://localhost:11434" # Default Ollama URL
LLM_TIMEOUT = 120 # Timeout for the LLM call

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
    Initializes and returns the ChromaDB client, SentenceTransformer model, and Ollama client.
    Using @st.cache_resource ensures this runs only once.
    """
    try:
        # 1. Initialize ChromaDB
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        
        # 2. Initialize Sentence Transformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
        # 3. Initialize Ollama Chat Model
        # LangChain handles the connection to the running Ollama server
        ollama_client = ChatOllama(
            base_url=OLLAMA_URL,
            model=OLLAMA_MODEL,
            temperature=0.7,
            request_timeout=LLM_TIMEOUT,
        )
        
        return db_client, model, ollama_client
    except Exception as e:
        st.error(f"An error occurred during dependency initialization. Error: {e}")
        st.stop()
        
def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

@traceable(run_type="llm")
def call_local_llm(prompt, max_retries=5):
    """
    Calls the local Ollama LLM via LangChain.
    The @traceable decorator creates an 'llm' run in LangSmith.
    """
    ollama_client = st.session_state.ollama_client
    
    system_message = SystemMessage(content="You are a helpful assistant. Be concise and accurate.")
    user_message = HumanMessage(content=prompt)
    
    # Simple retry mechanism for local service/network issues
    for i in range(max_retries):
        try:
            with st.spinner(f"Contacting {OLLAMA_MODEL} on {OLLAMA_URL}... (Attempt {i+1}/{max_retries})"):
                response = ollama_client.invoke([system_message, user_message])
            return response.content
            
        except requests.exceptions.ConnectionError:
            st.warning(f"Connection error to Ollama at {OLLAMA_URL}. Ensure Ollama is running and the model '{OLLAMA_MODEL}' is pulled. Retrying in 2 seconds...")
            time.sleep(2)
        except ValidationError as e:
            st.error(f"Pydantic Validation Error (often means malformed response or connection issue): {e}")
            return f"Error: Pydantic Validation Error during local LLM call."
        except Exception as e:
            st.error(f"An unexpected error occurred during the local LLM call: {e}")
            return f"Error: An unexpected error occurred: {e}"

    st.error("Failed to connect to Ollama after multiple retries. Please check your Ollama server.")
    return "Error: Failed to get response from local LLM."


def clear_chroma_data():
    """Clears all data from the ChromaDB collection."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            # Delete the current collection and create a new empty one
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
    The @traceable decorator creates a 'retriever' run in LangSmith.
    """
    collection = get_collection()
    model = st.session_state.model
    
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    # Filter out empty documents array if no results are found
    if results and results.get('documents') and results['documents'][0]:
        return results['documents'][0]
    return []

def is_valid_github_raw_url(url):
    """Simple check for a GitHub raw file URL for .txt or .md."""
    pattern = re.compile(r"^https://raw\.githubusercontent\.com/.*?\.(txt|md)$", re.IGNORECASE)
    return bool(pattern.match(url))

@traceable(run_type="chain")
def rag_pipeline(query, selected_language_code):
    """
    Executes the full RAG pipeline with a check for documents.
    The @traceable decorator creates a 'chain' run in LangSmith, which
    will nest the 'retriever' and 'llm' calls.
    """
    collection = get_collection()
    if collection.count() == 0:
        return "Hey there! I'm a chatbot that answers questions based on documents you provide. Please upload a `.txt`, `.pdf` file, or enter a GitHub raw URL in the section above before asking me anything. I'm ready when you are! 😊"

    # Calls the decorated retrieve_documents function (creates a nested 'retriever' run)
    relevant_docs = retrieve_documents(query)
    
    if not relevant_docs:
        # Fallback if the retriever returns nothing
        return "I couldn't find relevant information in the uploaded documents to answer your question."

    context = "\n".join(relevant_docs)
    
    # Ensure the prompt instructs the model to use the retrieved context and output the correct language
    prompt = (
        f"You are an expert document assistant. Using ONLY the 'Context' provided below, "
        f"answer the 'Question'. The final response MUST be in {st.session_state.selected_language}. "
        f"If the Context does not contain the answer, politely state that the information is missing. "
        f"\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    )
    
    # Calls the decorated call_local_llm function (creates a nested 'llm' run)
    response = call_local_llm(prompt)

    if response.startswith("Error:"):
        return response
    
    return response

def display_chat_messages():
    """Displays all chat messages in the Streamlit app."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    """Handles new user input, runs the RAG pipeline, and updates chat history."""
    if prompt := st.chat_input("Ask about your document..."):
        # Update chat history state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Run RAG and display assistant message
        with st.chat_message("assistant"):
            # A separate spinner is inside call_local_llm for better feedback
            selected_language_code = LANGUAGE_DICT[st.session_state.selected_language]
            # The LangSmith trace starts here for the RAG pipeline
            response = rag_pipeline(prompt, selected_language_code)
            st.markdown(response)

        # Update chat history with assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Update the chat title for the sidebar if it's the first message
        if st.session_state.current_chat_id and st.session_state.chat_history[st.session_state.current_chat_id]['title'] == "New Chat":
            # Use the first 50 chars of the user's first prompt as the title
            title = prompt[:50] + ('...' if len(prompt) > 50 else '')
            st.session_state.chat_history[st.session_state.current_chat_id]['title'] = title


# --- Streamlit UI ---
def main_ui():
    """Sets up the main Streamlit UI for the RAG chatbot."""
    st.set_page_config(
        page_title="RAG Chat Flow (Ollama)", 
        layout="wide",
        initial_sidebar_state="auto"
    )

    # Initialize dependencies: db_client, model, and the new ollama_client
    if 'db_client' not in st.session_state or 'model' not in st.session_state or 'ollama_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.ollama_client = initialize_dependencies()
        
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}
    
    if 'current_chat_id' not in st.session_state or not st.session_state.messages:
        # Start a new chat session if none exists or if the current one is empty
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
        st.markdown(f"Running LLM: **{OLLAMA_MODEL}** via Ollama at **{OLLAMA_URL}**")
        st.warning("Ensure Ollama is running and the model is pulled!")

        st.session_state.selected_language = st.selectbox(
            "Select Response Language",
            options=list(LANGUAGE_DICT.keys()),
            key="language_selector"
        )
        
        if st.button("New Chat / Clear Documents", use_container_width=True):
            st.session_state.messages = []
            clear_chroma_data() # Clear all documents on new chat
            # Reset chat history logic for a clean start
            st.session_state.chat_history = {}
            st.session_state.current_chat_id = None
            st.experimental_rerun()

        st.subheader("Chat History")
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            # Sort chats by date
            sorted_chat_ids = sorted(
                st.session_state.chat_history.keys(), 
                key=lambda x: st.session_state.chat_history[x]['date'], 
                reverse=True
            )
            for chat_id in sorted_chat_ids:
                chat_title = st.session_state.chat_history[chat_id]['title']
                date_str = st.session_state.chat_history[chat_id]['date'].strftime("%b %d, %I:%M %p")
                
                # Highlight the current chat
                is_current = chat_id == st.session_state.current_chat_id
                style = "background-color: #262730; border-radius: 5px; padding: 10px;" if is_current else "padding: 10px;"
                
                # Using a container for better click area
                with st.container():
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
    st.title("📚 Retrieval Augmented Generation (RAG) Chatbot - Local LLM")
    st.info("Upload documents (TXT or PDF) to provide context. Powered by Ollama/Mistral and traceable in LangSmith.")
    
    # Document upload/processing section
    with st.container():
        st.subheader("Add Context Documents")
        # Combined file uploader for TXT and PDF
        uploaded_files = st.file_uploader(
            "Upload files (.txt, .pdf)", 
            type=["txt", "pdf"], 
            accept_multiple_files=True
        )
        github_url = st.text_input("Enter a GitHub raw `.txt` or `.md` URL (e.g., https://raw.githubusercontent.com/user/repo/branch/file.txt):")

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
                                # New PDF handling logic
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
