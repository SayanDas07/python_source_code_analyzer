import streamlit as st
import os
import sys
import traceback
import logging
import shutil
from langchain.vectorstores import Chroma
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


from src.helper import load_embedding, repo_ingestion

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


try:
    load_dotenv()
    logger.info("Environment variables loaded successfully")
except Exception as env_error:
    logger.error(f"Error loading environment variables: {str(env_error)}")
    logger.error(traceback.format_exc())

if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'qa' not in st.session_state:
    st.session_state.qa = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'memory' not in st.session_state:
    st.session_state.memory = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'repo_loaded' not in st.session_state:
    st.session_state.repo_loaded = False

def initialize_llm():
    """Initialize the LLM and memory components"""
    try:
        logger.debug("Initializing LLM components")
        st.session_state.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.4,
            max_tokens=700
        )
        st.session_state.memory = ConversationSummaryMemory(
            llm=st.session_state.llm,
            memory_key="chat_history",
            return_messages=True
        )
        logger.info("LLM and memory components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def initialize_vectordb():
    """Initialize the vector database and QA components"""
    try:
        logger.debug("Initializing vector database")

        if st.session_state.llm is None or st.session_state.memory is None:
            if not initialize_llm():
                return False
        

        embeddings = load_embedding()
        persist_directory = "db"
      
        if not os.path.exists(persist_directory):
            logger.warning(f"Vector database directory '{persist_directory}' does not exist")
            return False
            
        dir_contents = os.listdir(persist_directory)
        if not dir_contents:
            logger.warning(f"Vector database directory '{persist_directory}' is empty")
            return False
            
        logger.debug(f"DB directory contents: {dir_contents}")
        

        st.session_state.vectordb = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings
        )
        

        st.session_state.qa = ConversationalRetrievalChain.from_llm(
            st.session_state.llm, 
            retriever=st.session_state.vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), 
            memory=st.session_state.memory
        )
        
        logger.info("Vector database and QA components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing vector database: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def clear_repo():
    """Clear the repository and database directories"""
    try:
        cleared = False
       
        if os.path.exists("repo"):
            try:
                logger.debug("Removing repository directory")
                shutil.rmtree("repo")
                logger.info("Removed repository directory")
                cleared = True
            except Exception as rm_error:
                logger.error(f"Failed to remove repo directory: {str(rm_error)}")
                logger.error(traceback.format_exc())
                st.error("Failed to clear repository directory")
                return False
        

        if os.path.exists("db"):
            try:
                logger.debug("Removing vector database directory")
                shutil.rmtree("db")
                logger.info("Removed vector database directory")
                cleared = True
            except Exception as rm_error:
                logger.error(f"Failed to remove db directory: {str(rm_error)}")
                logger.error(traceback.format_exc())
                st.error("Failed to clear vector database directory")
                return False
        
        st.session_state.vectordb = None
        st.session_state.qa = None
        

        try:
            os.makedirs("db", exist_ok=True)
            os.makedirs("repo", exist_ok=True)
        except Exception as mk_error:
            logger.warning(f"Failed to create empty directories: {str(mk_error)}")
            st.warning("Failed to create empty directories")
        
        return True
    except Exception as clear_error:
        logger.error(f"Error clearing repository: {str(clear_error)}")
        logger.error(traceback.format_exc())
        st.error("Error clearing repository data")
        return False

def load_repository(repo_url):
    """Load and process a GitHub repository"""
    with st.spinner("Processing repository..."):
        try:
            # PHASE 1: Clean up existing directories
            st.text("Cleaning up existing directories...")
            if not clear_repo():
                st.error("Failed to clean up existing directories")
                return False
            
            # PHASE 2: Clone the repository
            st.text(f"Cloning repository: {repo_url}")
            try:
                repo_ingestion(repo_url)
                logger.info(f"Repository {repo_url} cloned successfully")
            except Exception as repo_error:
                logger.error(f"Failed to clone repository: {str(repo_error)}")
                logger.error(traceback.format_exc())
                st.error(f"Failed to clone repository: {str(repo_error)}")
                return False
            
            # PHASE 3: Run indexing
            st.text("Indexing repository files...")
            try:
                exit_code = os.system("python store_index.py")
                if exit_code != 0:
                    logger.error(f"store_index.py failed with exit code {exit_code}")
                    st.error(f"Failed to process repository contents (exit code: {exit_code})")
                    return False
                logger.info("Repository indexing completed successfully")
            except Exception as index_error:
                logger.error(f"Error during indexing: {str(index_error)}")
                logger.error(traceback.format_exc())
                st.error(f"Failed during indexing: {str(index_error)}")
                return False
            
            # PHASE 4: Initialize components
            st.text("Initializing components...")
            try:
               
                if st.session_state.llm is None or st.session_state.memory is None:
                    if not initialize_llm():
                        st.error("Failed to initialize LLM components")
                        return False
                
       
                if not initialize_vectordb():
                    st.error("Repository indexed but failed to initialize search capabilities")
                    return False
                    
                logger.info("All components initialized successfully")
                st.session_state.repo_loaded = True
                return True
            except Exception as init_error:
                logger.error(f"Error during component initialization: {str(init_error)}")
                logger.error(traceback.format_exc())
                st.error(f"Failed during component initialization: {str(init_error)}")
                return False
                
        except Exception as e:
            logger.critical(f"Unhandled error in load_repository: {str(e)}")
            logger.critical(traceback.format_exc())
            st.error(f"An unexpected error occurred: {str(e)}")
            return False


def process_question(question):
    """Process a user question using the QA chain"""
    try:
        if st.session_state.qa is None:
            return "Please load a repository first"
            
        logger.debug("Processing message with QA chain")
        result = st.session_state.qa(question)
        logger.info("Message processed successfully")
        return result["answer"]
    except Exception as qa_error:
        logger.error(f"Error in QA processing: {str(qa_error)}")
        logger.error(traceback.format_exc())
        return "Sorry, I had trouble processing your question. Please try again or check the logs for details."

st.title("GitHub Repository Analyzer")
st.write("Load a GitHub repository and ask questions about the code.")


with st.sidebar:
    st.header("Repository Management")
    repo_url = st.text_input("Enter GitHub Repository URL", placeholder="https://github.com/username/repo")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Repository"):
            if repo_url and repo_url.strip():
                success = load_repository(repo_url)
                if success:
                    st.success(f"Repository {repo_url} loaded successfully")
                    st.session_state.chat_history = [] 
            else:
                st.error("Repository URL cannot be empty")
    
    with col2:
        if st.button("Clear Repository"):
            if clear_repo():
                st.success("Repository data cleared successfully")
                st.session_state.repo_loaded = False
                st.session_state.chat_history = []  

    st.divider()
    st.write("Status: " + ("Repository loaded" if st.session_state.repo_loaded else "No repository loaded"))


st.header("Chat with your Repository")


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input("Ask a question about the repository..." if st.session_state.repo_loaded else "Load a repository first..."):

    st.session_state.chat_history.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.write(user_question)
    
  
    with st.chat_message("assistant"):
        if st.session_state.repo_loaded:
            response = process_question(user_question)
        else:
            response = "Please load a repository first using the sidebar."
        
        st.write(response)
    
  
    st.session_state.chat_history.append({"role": "assistant", "content": response})


if os.path.exists("db") and os.path.exists("repo"):
    dir_contents = os.listdir("db")
    if dir_contents:
        if initialize_llm() and initialize_vectordb():
            st.session_state.repo_loaded = True