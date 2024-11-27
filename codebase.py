"""
Project: 
PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

Description:
Allow users to upload a PDF and then ask questions about the content using a local model.
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

# Set protobuf environment variable to avoid error messages
# This might cause some issues with latency but it's a tradeoff
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=True)
def extract_model_names(
    models_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info (Dict[str, List[Dict[str, Any]]]): Dictionary containing information about available models.

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    model_names = tuple(model["name"] for model in models_info["models"])
    logger.info(f"Extracted model names: {model_names}")
    return model_names


def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.


    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.


    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

        # Create layout
    col1, col2 = st.columns([1.5, 2])


    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False


    # Model selection
    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", 
            available_models,
            key="model_select"
        )


    # Add checkbox for sample PDF
    use_sample = col1.toggle(
        "Use sample PDF (Scammer Agent Paper)", 
        key="sample_checkbox"
    )
    
    # Clear vector DB if switching between sample and upload
    if use_sample != st.session_state.get("use_sample"):
        if st.session_state["vector_db"] is not None:
            st.session_state["vector_db"].delete_collection()
            st.session_state["vector_db"] = None
            st.session_state["pdf_pages"] = None
        st.session_state["use_sample"] = use_sample


    if use_sample:
        # Use the sample PDF
        sample_path = "scammer-agent.pdf"
        if os.path.exists(sample_path):
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing sample PDF..."):
                    loader = UnstructuredPDFLoader(file_path=sample_path)
                    data = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                    chunks = text_splitter.split_documents(data)
                    st.session_state["vector_db"] = Chroma.from_documents(
                        documents=chunks,
                        embedding=OllamaEmbeddings(model="nomic-embed-text"),
                        collection_name="myRAG"
                    )
                    # Open and display the sample PDF
                    with pdfplumber.open(sample_path) as pdf:
                        pdf_pages = [page.to_image().original for page in pdf.pages]
                        st.session_state["pdf_pages"] = pdf_pages
        else:
            st.error("Sample PDF file not found in the current directory.")
    else:
        # Regular file upload with unique key
        file_upload = col1.file_uploader(
            "Upload a PDF file ↓", 
            type="pdf", 
            accept_multiple_files=False,
            key="pdf_uploader"
        )


        if file_upload:
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing uploaded PDF..."):
                    st.session_state["vector_db"] = create_vector_db(file_upload)
                    pdf_pages = extract_all_pages_as_images(file_upload)
                    st.session_state["pdf_pages"] = pdf_pages


    # Display PDF if pages are available
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        # PDF display controls
        zoom_level = col1.slider(
            "Zoom Level", 
            min_value=100, 
            max_value=1000, 
            value=700, 
            step=50,
            key="zoom_slider"
        )


        # Display PDF pages
        with col1:
            with st.container(height=410, border=True):
                # Removed the key parameter from st.image()
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)


    # Delete collection button
    delete_collection = col1.button(
        "⚠️ Delete collection", 
        type="secondary",
        key="delete_button"
    )


    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])


    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)


        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])


        # Chat input and processing
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)


                # Process and display assistant response
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")


                # Add assistant response to chat history
                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )


            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file or use the sample PDF to begin chat...")




if __name__ == "__main__":
    main()