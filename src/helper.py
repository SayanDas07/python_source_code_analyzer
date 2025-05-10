import os
import logging
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError
import traceback
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def repo_ingestion(repo_url):
    """
    Clone a Git repository from the provided URL.
    
    Args:
        repo_url (str): URL of the Git repository to clone
        
    Returns:
        None
        
    Raises:
        ValueError: If repo_url is empty or invalid
        GitCommandError: If Git clone operation fails
    """
    if not repo_url or not repo_url.strip():
        raise ValueError("Repository URL cannot be empty")
    
    logger.info(f"Cloning repository from: {repo_url}")
    
    # Create directory if it doesn't exist
    try:
        os.makedirs("repo", exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory 'repo': {str(e)}")
        raise
    
    repo_path = "repo/"
    
    # Check if directory already contains a repository
    try:
        if os.path.exists(os.path.join(repo_path, ".git")):
            logger.warning(f"Repository already exists at {repo_path}. Removing it before cloning.")
            import shutil
            try:
                shutil.rmtree(repo_path)
                os.makedirs(repo_path, exist_ok=True)
            except OSError as rm_error:
                logger.error(f"Failed to remove existing repository: {str(rm_error)}")
                raise
    except Exception as check_error:
        logger.error(f"Error checking existing repository: {str(check_error)}")
        raise
    
    # Clone the repository
    try:
        Repo.clone_from(repo_url, to_path=repo_path)
        logger.info(f"Repository cloned successfully to {repo_path}")
    except GitCommandError as git_error:
        logger.error(f"Git clone failed: {str(git_error)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during repository cloning: {str(e)}")
        raise


def load_repo(repo_path):
    """
    Load Python documents from a repository.
    
    Args:
        repo_path (str): Path to the repository
        
    Returns:
        list: List of loaded documents
        
    Raises:
        FileNotFoundError: If repo_path doesn't exist
        Exception: For other document loading errors
    """
    if not os.path.exists(repo_path):
        logger.error(f"Repository path does not exist: {repo_path}")
        raise FileNotFoundError(f"Repository path not found: {repo_path}")
        
    logger.info(f"Loading documents from {repo_path}")
    
    try:
        loader = GenericLoader.from_filesystem(
            repo_path,
            glob="**/*",
            suffixes=[".py"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
        )
        
        documents = loader.load()
        
        if not documents:
            logger.warning(f"No Python documents found in {repo_path}")
        else:
            logger.info(f"Loaded {len(documents)} Python documents")
            
        return documents
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def text_splitter(documents):
    """
    Split documents into smaller text chunks.
    
    Args:
        documents (list): List of documents to split
        
    Returns:
        list: List of text chunks
        
    Raises:
        ValueError: If documents is empty
        Exception: For other text splitting errors
    """
    if not documents:
        logger.error("No documents provided for splitting")
        raise ValueError("Cannot split empty document list")
        
    logger.info(f"Splitting {len(documents)} documents into chunks")
    
    try:
        documents_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=300,
            chunk_overlap=20
        )
        
        text_chunks = documents_splitter.split_documents(documents)
        
        logger.info(f"Created {len(text_chunks)} text chunks")
        return text_chunks
        
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def load_embedding():
    """
    Load Google Generative AI embeddings.
    
    Returns:
        GoogleGenerativeAIEmbeddings: Embedding model
        
    Raises:
        Exception: If loading embeddings fails
    """
    logger.info("Loading Google Generative AI embeddings")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        raise ValueError("GOOGLE_API_KEY environment variable is required but not set")
        
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        logger.info("Embeddings loaded successfully")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        logger.error(traceback.format_exc())
        raise