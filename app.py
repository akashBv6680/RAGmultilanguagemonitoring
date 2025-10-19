import streamlit as st
import os
import time
import json
import uuid
from google import genai
from google.genai.errors import APIError

# --- Third-party library for RAG components (LangChain and Chroma) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Firebase Libraries (Used for Chat History) ---
# NOTE: We use placeholders for Firebase imports as they are handled by the runtime environment
# but include logic to check for the global variables (__app_id, etc.)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth, exceptions
except ImportError:
    st.error("Firebase Admin SDK not found. Please ensure it is installed if running locally.")
    firebase_admin = None
    firestore = None
    auth = None

# --- Global Configuration and Setup ---

# MANDATORY: Access the global variables provided by the Canvas environment
APP_ID = os.environ.get('__app_id', 'default-rag-app')
FIREBASE_CONFIG = os.environ.get('__firebase_config')
INITIAL_AUTH_TOKEN = os.environ.get('__initial_auth_token')

# Hardcoded RAG configuration
VECTOR_DB_PATH = "./rag_vector_db"
EMBEDDING_MODEL = 'text-embedding-004'
GENERATION_MODEL = 'gemini-2.5-flash-preview-09-2025'

# Multilingual support configuration
# EXPANDED LANGUAGE DICTIONARY
LANGUAGES = {
    "English": "Respond in clear, professional English.",
    "Spanish (Español)": "Respond in Spanish (Español) and maintain a formal, polite tone.",
    "Arabic (العربية)": "Respond in Arabic (العربية) and maintain a formal, polite tone.",
    "French (Français)": "Respond in French (Français) and maintain a formal, polite tone.",
    "German (Deutsch)": "Respond in German (Deutsch) and maintain a formal, polite tone.",
    "Hindi (हिंदी)": "Respond in Hindi (हिंदी) and maintain a formal, polite tone.",
    "Tamil (தமிழ்)": "Respond in Tamil (தமிழ்) and maintain a formal, polite tone.",
    "Bengali (বাংলা)": "Respond in Bengali (বাংলা) and maintain a formal, polite tone.",
    "Japanese (日本語)": "Respond in Japanese (日本語) and maintain a formal, polite tone.",
    "Korean (한국어)": "Respond in Korean (한국어) and maintain a formal, polite tone.",
    "Russian (русский)": "Respond in Russian (русский) and maintain a formal, polite tone.",
    "Chinese (Simplified) (简体中文)": "Respond in Chinese (Simplified) (简体中文) and maintain a formal, polite tone.",
    "Portuguese (Português)": "Respond in Portuguese (Português) and maintain a formal, polite tone.",
    "Italian (Italiano)": "Respond in Italian (Italiano) and maintain a formal, polite tone.",
    "Dutch (Nederlands)": "Respond in Dutch (Nederlands) and maintain a formal, polite tone.",
    "Turkish (Türkçe)": "Respond in Turkish (Türkçe) and maintain a formal, polite tone.",
}

# --- Core Firebase/Firestore Functions ---

@st.cache_resource
def init_firestore_and_auth():
    """Initializes Firebase and authenticates the user."""
    if not firebase_admin:
        st.error("Firebase Admin SDK is required but failed to import.")
        return None, None, 'anonymous'

    if not FIREBASE_CONFIG:
        st.error("Firebase config not found.")
        return None, None, 'anonymous'

    try:
        config = json.loads(FIREBASE_CONFIG)
        if not firebase_admin._apps:
            cred = credentials.Certificate(config)
            firebase_admin.initialize_app(cred, name='rag_app')
        db = firestore.client(firebase_admin.get_app('rag_app'))
        auth_client = auth.Client(firebase_admin.get_app('rag_app'))

        # Authenticate with custom token or anonymously
        user_id = 'anonymous'
        if INITIAL_AUTH_TOKEN:
            try:
                # Sign in with custom token and get UID
                decoded_token = auth_client.verify_id_token(INITIAL_AUTH_TOKEN)
                user_id = decoded_token['uid']
            except exceptions.FirebaseError as e:
                st.warning(f"Custom auth failed: {e}. Falling back to anonymous.")
                user_id = 'anonymous-' + str(uuid.uuid4())
        else:
            user_id = 'anonymous-' + str(uuid.uuid4())

        return db, auth_client, user_id

    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        return None, None, 'anonymous'

def get_chat_history_ref(db, user_id):
    """Returns the Firestore reference for the user's chat history collection."""
    if not db:
        return None
    # Storage path: /artifacts/{appId}/users/{userId}/rag_chat_history
    return db.collection('artifacts').document(APP_ID).collection('users').document(user_id).collection('rag_chat_history')

def load_chat_history(db, user_id):
    """Loads chat history from Firestore."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if db and user_id != 'anonymous':
        try:
            history_ref = get_chat_history_ref(db, user_id)
            query = history_ref.order_by('timestamp')
            docs = query.stream()
            
            # Only load if session state is empty (first load)
            if not st.session_state.messages:
                for doc in docs:
                    data = doc.to_dict()
                    st.session_state.messages.append({"role": data['role'], "content": data['content']})

        except Exception as e:
            st.error(f"Error loading chat history: {e}")

def save_chat_message(db, user_id, role, content):
    """Saves a single message to Firestore."""
    if db and user_id != 'anonymous':
        try:
            history_ref = get_chat_history_ref(db, user_id)
            history_ref.add({
                'role': role,
                'content': content,
                'timestamp': firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            st.error(f"Error saving chat message: {e}")

# --- API and RAG Helper Functions ---

# FIX: This function explicitly handles the 'NoneType' error by checking the response structure
def safe_extract_response_text(api_response):
    """
    Safely extracts the text content from the Gemini API response candidate,
    preventing the 'NoneType' object has no attribute 'strip' error.
    """
    if (
        api_response and
        api_response.candidates and
        api_response.candidates[0].content and
        api_response.candidates[0].content.parts and
        api_response.candidates[0].content.parts[0].text
    ):
        response_text = api_response.candidates[0].content.parts[0].text
        # Ensure it's treated as a string before strip is called
        return str(response_text).strip()
    else:
        # Check for safety filter failure (a common reason for None/empty response)
        try:
            safety_ratings = api_response.candidates[0].safety_ratings
            reasons = ", ".join([r.probability.name for r in safety_ratings if r.blocked])
            st.warning(f"Content generation failed due to safety filters. Blocked reasons: {reasons}")
        except:
            pass # Ignore if safety ratings structure is also missing

        # Return a non-None, user-friendly error message
        return "I am unable to generate a response for this query or language. Please try again or rephrase your request."

@st.cache_resource
def get_vector_store():
    """Initializes and returns the Chroma vector store."""
    try:
        if "GOOGLE_API_KEY" not in os.environ:
            st.error("API Key not configured in environment.")
            return None

        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        # Use a temporary directory for Chroma storage
        return Chroma(
            collection_name="rag_documents",
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        return None

def process_documents(uploaded_file, vector_store):
    """Loads PDF, chunks text, and generates embeddings into the vector store."""
    if not vector_store:
        st.error("Vector store is not initialized.")
        return False

    with open(os.path.join("/tmp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 1. Load the document
    loader = PyPDFLoader(os.path.join("/tmp", uploaded_file.name))
    documents = loader.load()

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # 3. Add chunks to ChromaDB (Embedding is handled internally by the embedding function)
    vector_store.add_documents(chunks)
    vector_store.persist() # Save the database state
    return True

def generate_response(query, vector_store, chat_history):
    """
    Retrieves context from the vector store and generates a final response using Gemini.
    """
    if not vector_store:
        yield "Error: RAG components not ready."
        return

    try:
        # 1. Retrieval: Get relevant context chunks
        context_docs = vector_store.similarity_search(query, k=4)
        context = "\n---\n".join([doc.page_content for doc in context_docs])

        # 2. Build the System Prompt
        system_prompt = f"""
        You are a Document-Grounded Assistant. Your primary goal is to answer the user's question
        based **only** on the context provided below. If the answer is not in the context,
        state clearly that you cannot find the information in the provided documents.

        CONTEXT:
        ---
        {context}
        ---

        USER CHAT HISTORY (for reference):
        {json.dumps(chat_history)}

        {st.session_state.language_prompt_instruction}
        """

        # 3. Call the Gemini API
        client = genai.Client()
        response_stream = client.models.generate_content_stream(
            model=GENERATION_MODEL,
            contents=query,
            config={"system_instruction": system_prompt}
        )

        full_response = ""
        for chunk in response_stream:
            if chunk.text:
                full_response += chunk.text
                yield chunk.text

        # Use the safe extractor after the stream to get the final, complete text
        # (This is more complex with streaming, so we'll ensure the final accumulated string is safe)
        final_text = full_response
        if not final_text:
             # If stream yields nothing, attempt a non-streaming call to check for errors/None
             response_non_stream = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=query,
                config={"system_instruction": system_prompt}
            )
             final_text = safe_extract_response_text(response_non_stream)
             if not final_text or final_text.startswith("I am unable to generate"):
                yield final_text # Yield the error message

        # After streaming, return the final response text
        return final_text

    except APIError as e:
        st.error(f"Gemini API Error: {e}")
        yield f"An API error occurred: {e}"
    except Exception as e:
        st.error(f"An unexpected error occurred during generation: {e}")
        yield f"An unexpected error occurred: {e}"


# --- Streamlit Application ---

# Initialize state variables
if "db" not in st.session_state:
    st.session_state.db, st.session_state.auth_client, st.session_state.user_id = init_firestore_and_auth()
if "language_prompt_instruction" not in st.session_state:
    st.session_state.language_prompt_instruction = LANGUAGES["English"]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store()
if "rag_loaded" not in st.session_state:
    # Check if the vector store already has content on startup
    try:
        st.session_state.rag_loaded = st.session_state.vector_store.get(include=["metadatas"])['metadatas'] != []
    except:
        st.session_state.rag_loaded = False

# Load chat history from Firestore on first run
load_chat_history(st.session_state.db, st.session_state.user_id)


st.title("📄 Multilingual RAG Chatbot (Gemini + Streamlit)")
st.caption(f"User ID: `{st.session_state.user_id}`")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("1. Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF for RAG",
        type="pdf",
        accept_multiple_files=False,
        disabled=not st.session_state.vector_store
    )

    if uploaded_file:
        if st.button("Process Document & Load RAG"):
            with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
                if process_documents(uploaded_file, st.session_state.vector_store):
                    st.session_state.rag_loaded = True
                    st.success("Document loaded and embeddings generated!")
                else:
                    st.error("Failed to process document.")

    if st.session_state.rag_loaded:
        st.success("✅ RAG Document Store is ready!")
    else:
        st.warning("Upload and process a PDF to enable RAG.")

    st.header("2. Language Selection")
    selected_lang = st.selectbox(
        "Choose Bot Response Language",
        options=list(LANGUAGES.keys()),
        index=0 # Default to English
    )
    st.session_state.language_prompt_instruction = LANGUAGES[selected_lang]
    st.info(f"The bot will respond in: **{selected_lang}**")

# --- Main Chat Interface ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your document..."):
    # 1. Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_message(st.session_state.db, st.session_state.user_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check RAG status
    if not st.session_state.rag_loaded:
        with st.chat_message("assistant"):
            st.warning("Please upload and process a document in the sidebar first.")
            st.stop()

    # 2. Get and stream assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Generator for streaming the response
        response_generator = generate_response(
            prompt,
            st.session_state.vector_store,
            st.session_state.messages
        )

        for chunk in response_generator:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌") # Streaming effect
        
        message_placeholder.markdown(full_response) # Final output

    # 3. Add assistant response to state and save
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_chat_message(st.session_state.db, st.session_state.user_id, "assistant", full_response)
