from src.helper import load_repo, text_splitter, load_embedding
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import os
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    logger.info("Loading environment variables")
    load_dotenv()

    # Check if repo directory exists
    if not os.path.exists("repo/"):
        logger.error("Repository directory does not exist. Please clone a repository first.")
        sys.exit(1)
    
    # Check if repo directory is empty
    if not os.listdir("repo/"):
        logger.error("Repository directory is empty. Please clone a repository first.")
        sys.exit(1)

    logger.info("Loading repository documents")
    try:
        documents = load_repo("repo/")
        if not documents:
            logger.warning("No Python documents found in the repository")
            sys.exit(1)
        logger.info(f"Loaded {len(documents)} documents from repository")
    except Exception as doc_error:
        logger.error(f"Error loading repository documents: {str(doc_error)}")
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Splitting text into chunks")
    try:
        text_chunks = text_splitter(documents)
        logger.info(f"Created {len(text_chunks)} text chunks")
    except Exception as split_error:
        logger.error(f"Error splitting text: {str(split_error)}")
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Loading embeddings")
    try:
        embeddings = load_embedding()
    except Exception as embed_error:
        logger.error(f"Error loading embeddings: {str(embed_error)}")
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Creating and storing vector database")
    try:
        # Make sure the db directory exists
        os.makedirs('./db', exist_ok=True)
        
        # Create and persist the vector database
        vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./db')
        vectordb.persist()
        logger.info("Vector database created and persisted successfully")
    except Exception as db_error:
        logger.error(f"Error creating vector database: {str(db_error)}")
        traceback.print_exc()
        sys.exit(1)
        
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

logger.info("Process completed successfully")