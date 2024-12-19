import os
import streamlit as st
import shutil
from dotenv import load_dotenv
from embed import embed
from query import query
from get_vector_db import get_vector_db
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
os.makedirs(TEMP_FOLDER, exist_ok=True)

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        if os.path.exists(TEMP_FOLDER):
            files = os.listdir(TEMP_FOLDER)
            for file in files:
                file_path = os.path.join(TEMP_FOLDER, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f"Error cleaning up {file_path}: {str(e)}")
                    st.error(f"Error cleaning up {file_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        st.error(f"Error during cleanup: {str(e)}")

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

def main():
    st.title("Document Query System")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["Embed Document", "Query", "Manage Database", "Cleanup"])
    
    # Embed Document Tab
    with tab1:
        st.header("Upload and Embed Document")
        uploaded_file = st.file_uploader("Choose a file to embed", type=['txt', 'pdf', 'doc', 'docx'])
        uploaded_file_name = uploaded_file.name if uploaded_file else None
        if uploaded_file is not None:
            #Now it is safe to access file attributes
            file_name = uploaded_file.name
            file_type = uploaded_file.type
            file_size = uploaded_file.size
            st.write("File name:", file_name)
            st.write("File type:", file_type)
            st.write("File size:", file_size, "bytes")
            if st.button("Embed Document"):
                with st.spinner("Embedding document..."):
                    try:
                        # The embed function processes the uploaded file and stores its embeddings in the database.
                        # Expected input: uploaded_file (a file-like object)
                        # Expected output: success (boolean indicating if the embedding was successful)
                        success = embed(uploaded_file)
                        if success:
                            st.session_state.processed_files.add(uploaded_file.name)
                            st.success("File embedded successfully!")
                        else:
                            st.error("Failed to embed file")
                    except Exception as e:
                        st.error(f"Error during embedding: {str(e)}")
                
    
    # Query Tab
    with tab2:
        st.header("Query Documents")
        user_query = st.text_area("Enter your query:")
        
        if st.button("Submit Query"):
            if user_query:
                with st.spinner("Processing query..."):
                    try:
                        response = query(user_query)
                        if response:
                            st.write("Response:")
                            st.write(response)
                        else:
                            st.error("Failed to process query")
                    except Exception as e:
                        logger.error(f"Error during query: {str(e)}")
                        st.error(f"Error during query: {str(e)}")
            else:
                st.warning("Please enter a query")
    
    # Database Management Tab
    with tab3:
        st.header("Database Management")
        st.warning("Warning: This will permanently delete all embedded documents!")
        if st.button("Delete Collection", type="secondary"):
            try:
                db = get_vector_db()
                db.delete_collection()
                st.session_state.processed_files.clear()
                cleanup_temp_files()  # Clean up after deletion
                st.success("Collection deleted successfully")
            except Exception as e:
                logger.error(f"Error deleting collection: {str(e)}")
                st.error(f"Error deleting collection: {str(e)}")
    
    # Cleanup Tab
    with tab4:
        st.header("System Cleanup")
        st.write("Processed files in this session:", st.session_state.processed_files)
        
        if st.button("Clean Temporary Files"):
            with st.spinner("Cleaning up temporary files..."):
                cleanup_temp_files()
                st.success("Temporary files cleaned up successfully")

        if st.button("Reset Session"):
            st.session_state.processed_files.clear()
            cleanup_temp_files()
            st.success("Session reset successfully")

if __name__ == "__main__":
    main()