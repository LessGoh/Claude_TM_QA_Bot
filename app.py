import streamlit as st
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Page configuration
st.set_page_config(
    page_title="PDF QA Bot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'user_session_id' not in st.session_state:
    st.session_state.user_session_id = str(uuid.uuid4())[:8]

def get_api_keys():
    """Get API keys from Streamlit secrets or user input"""
    openai_key = None
    pinecone_key = None
    
    # Try to get from secrets first
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
        pinecone_key = st.secrets["PINECONE_API_KEY"]
        return openai_key, pinecone_key, True
    except (KeyError, FileNotFoundError):
        # Fall back to user input
        return None, None, False

def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)

def setup_pinecone(pinecone_key):
    """Setup Pinecone vector database with persistent index"""
    try:
        os.environ["PINECONE_API_KEY"] = pinecone_key
        pc = Pinecone(api_key=pinecone_key)
        spec = ServerlessSpec(cloud='aws', region='us-east-1')
        
        # Use persistent index name
        index_name = "pdf-qa-permanent"
        existing_indexes = [item['name'] for item in pc.list_indexes().indexes]
        
        # Create index only if it doesn't exist
        if index_name not in existing_indexes:
            # Check if we've hit the index limit
            if len(existing_indexes) >= 5:
                st.error("⚠️ Pinecone index limit reached (5/5). Please delete unused indexes or upgrade your plan.")
                st.write("**Existing indexes:**")
                for idx in existing_indexes:
                    st.write(f"- {idx}")
                
                # Offer to delete old qa-bot indexes
                old_indexes = [idx for idx in existing_indexes if idx.startswith('qa-bot-')]
                if old_indexes:
                    st.write("**Found old session indexes that can be safely deleted:**")
                    if st.button("🗑️ Delete old session indexes"):
                        deleted_count = 0
                        for old_idx in old_indexes:
                            try:
                                pc.delete_index(old_idx)
                                deleted_count += 1
                                st.write(f"✅ Deleted: {old_idx}")
                            except Exception as e:
                                st.write(f"❌ Failed to delete {old_idx}: {str(e)}")
                        
                        if deleted_count > 0:
                            st.success(f"Deleted {deleted_count} old indexes. Please refresh the page to continue.")
                            st.rerun()
                
                return None, None
            
            pc.create_index(
                index_name,
                dimension=1536,  # dimensionality of text-embedding-ada-002
                metric='cosine',
                spec=spec
            )
            
            # Wait for index to be ready
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        
        index = pc.Index(index_name)
        
        # Initialize embeddings with explicit parameters
        try:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=os.environ.get("OPENAI_API_KEY")
            )
        except Exception as e:
            st.error(f"Error initializing OpenAI embeddings: {str(e)}")
            return None, None
        
        vectorstore = PineconeVectorStore(index, embeddings, "text")
        return vectorstore, index_name
    
    except Exception as e:
        # Check if it's a quota error
        if "max serverless indexes" in str(e).lower() or "forbidden" in str(e).lower():
            st.error("🚫 **Pinecone Index Limit Reached**")
            st.write("You've reached the maximum number of indexes (5) allowed in the free plan.")
            st.write("**Solutions:**")
            st.write("1. Delete unused indexes from your Pinecone dashboard")
            st.write("2. Upgrade to a paid Pinecone plan")
            st.write("3. Use the 'Delete old session indexes' button if available")
        else:
            st.error(f"Error setting up Pinecone: {str(e)}")
        return None, None

def process_pdf(uploaded_file, vectorstore):
    """Process uploaded PDF file"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load and process PDF
        pdf_loader = PyPDFLoader(tmp_file_path)
        pages = pdf_loader.load_and_split()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(pages)
        
        # Add documents to vector store
        vectorstore.add_documents(documents=splits)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return True, len(splits)
    
    except Exception as e:
        return False, str(e)

def process_multiple_pdfs(uploaded_files, vectorstore):
    """Process multiple PDF files with error handling for each file"""
    import time
    
    def process_single_file(file_data):
        """Process a single PDF file"""
        file, vectorstore = file_data
        start_time = time.time()
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load and process PDF
            pdf_loader = PyPDFLoader(tmp_file_path)
            pages = pdf_loader.load_and_split()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(pages)
            
            # Add documents to vector store (thread-safe)
            vectorstore.add_documents(documents=splits)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            processing_time = time.time() - start_time
            return {
                "file": file.name,
                "status": "success", 
                "chunks": len(splits),
                "processing_time": processing_time
            }
            
        except Exception as e:
            # Clean up temporary file if it exists
            try:
                if 'tmp_file_path' in locals():
                    os.unlink(tmp_file_path)
            except:
                pass
            
            processing_time = time.time() - start_time
            return {
                "file": file.name,
                "status": "error",
                "error": str(e),
                "chunks": 0,
                "processing_time": processing_time
            }
    
    # Process files with progress tracking
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Prepare file data for processing
        file_data = [(file, vectorstore) for file in uploaded_files]
        
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, data): data[0].name 
            for data in file_data
        }
        
        # Process completed tasks
        completed = 0
        total_files = len(uploaded_files)
        
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            completed += 1
            
            # Update progress
            progress = completed / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing {filename}... ({completed}/{total_files})")
            
            # Get result
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "file": filename,
                    "status": "error",
                    "error": str(e),
                    "chunks": 0
                })
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return results

def ask_question(question, vectorstore):
    """Get answer to question using RAG"""
    try:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 relevant chunks
        )
        
        prompt_rag = PromptTemplate.from_template(
            "Answer the question: {question} based on the following context: {context}. "
            "Provide references to the source material when possible. "
            "If the answer cannot be found in the context, say so clearly."
        )
        
        llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0,
            max_tokens=1000
        )
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_rag
            | llm
        )
        
        result = rag_chain.invoke(question)
        return result.content
    
    except Exception as e:
        return f"Error processing question: {str(e)}"

def cleanup_resources():
    """Cleanup session resources (preserves persistent index)"""
    try:
        # Clean up any temporary files or session data
        # Index is preserved for future sessions
        if 'vectorstore' in st.session_state:
            st.session_state.vectorstore = None
        if 'documents_processed' in st.session_state:
            st.session_state.documents_processed = False
        if 'chat_history' in st.session_state:
            st.session_state.chat_history = []
    except:
        pass  # Silently fail cleanup

def clear_pinecone_index(pinecone_key):
    """Optional function to clear the persistent index (admin use)"""
    try:
        pc = Pinecone(api_key=pinecone_key)
        index_name = "pdf-qa-permanent"
        existing_indexes = [item['name'] for item in pc.list_indexes().indexes]
        
        if index_name in existing_indexes:
            index = pc.Index(index_name)
            index.delete(delete_all=True)
            return {"status": "success", "message": "Index cleared successfully"}
        else:
            return {"status": "warning", "message": "Index does not exist"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_index_stats(pinecone_key):
    """Get statistics about the Pinecone index"""
    try:
        pc = Pinecone(api_key=pinecone_key)
        index_name = "pdf-qa-permanent"
        existing_indexes = [item['name'] for item in pc.list_indexes().indexes]
        
        if index_name in existing_indexes:
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            return {
                "status": "success",
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
        else:
            return {"status": "warning", "message": "Index does not exist"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def list_all_indexes(pinecone_key):
    """List all indexes in the Pinecone account"""
    try:
        pc = Pinecone(api_key=pinecone_key)
        indexes = [item['name'] for item in pc.list_indexes().indexes]
        return {"status": "success", "indexes": indexes}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def delete_index(pinecone_key, index_name):
    """Delete a specific index"""
    try:
        pc = Pinecone(api_key=pinecone_key)
        pc.delete_index(index_name)
        return {"status": "success", "message": f"Index '{index_name}' deleted successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Main app
def main():
    st.title("📄 PDF Question Answering System")
    st.markdown("Upload a PDF document and ask questions about its content using AI.")
    
    # Get API keys
    openai_key, pinecone_key, using_secrets = get_api_keys()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        if not using_secrets:
            st.warning("API keys not found in secrets. Please enter them below:")
            
            openai_key = st.text_input(
                "🔑 OpenAI API Key", 
                type="password",
                help="Get your API key from https://platform.openai.com/api-keys"
            )
            
            pinecone_key = st.text_input(
                "🌲 Pinecone API Key", 
                type="password",
                help="Get your API key from https://app.pinecone.io/"
            )
        else:
            st.success("✅ API keys loaded from secrets")
        
        # Session info
        st.markdown("---")
        st.markdown(f"**Session ID:** `{st.session_state.user_session_id}`")
        st.markdown("**Index:** `pdf-qa-permanent` (persistent)")
        
        # Index statistics
        if pinecone_key:
            if st.button("📊 Show Index Stats"):
                with st.spinner("Getting index statistics..."):
                    stats = get_index_stats(pinecone_key)
                    if stats["status"] == "success":
                        st.metric("Total Documents", f"{stats['total_vectors']:,}")
                        st.metric("Index Fullness", f"{stats['index_fullness']:.2%}")
                    else:
                        st.warning(stats.get("message", "Could not get stats"))
        
        if st.button("🔄 Reset Session"):
            # Cleanup current session
            cleanup_resources()
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Admin functions
        st.markdown("---")
        st.markdown("**🔧 Admin Functions**")
        
        col_admin1, col_admin2 = st.columns(2)
        
        with col_admin1:
            if st.button("🗑️ Clear Index", help="Clear all documents from persistent index"):
                if pinecone_key:
                    with st.spinner("Clearing index..."):
                        result = clear_pinecone_index(pinecone_key)
                        if result["status"] == "success":
                            st.success(result["message"])
                            cleanup_resources()
                        elif result["status"] == "warning":
                            st.warning(result["message"])
                        else:
                            st.error(result["message"])
                else:
                    st.error("Pinecone API key required")
        
        with col_admin2:
            if st.button("📋 Manage Indexes", help="View and manage all Pinecone indexes"):
                if pinecone_key:
                    with st.spinner("Loading indexes..."):
                        result = list_all_indexes(pinecone_key)
                        if result["status"] == "success":
                            st.write(f"**Total indexes: {len(result['indexes'])}/5**")
                            for idx in result["indexes"]:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    if idx.startswith('qa-bot-'):
                                        st.write(f"🔄 {idx} (old session)")
                                    elif idx == 'pdf-qa-permanent':
                                        st.write(f"📌 {idx} (current)")
                                    else:
                                        st.write(f"📦 {idx}")
                                with col2:
                                    if idx != 'pdf-qa-permanent' and st.button("Delete", key=f"del_{idx}"):
                                        del_result = delete_index(pinecone_key, idx)
                                        if del_result["status"] == "success":
                                            st.success(f"Deleted {idx}")
                                            st.rerun()
                                        else:
                                            st.error(del_result["message"])
                        else:
                            st.error(result["message"])
                else:
                    st.error("Pinecone API key required")
    
    # Main content
    if openai_key and pinecone_key:
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = openai_key
        
        # Initialize vector store if not done
        if not st.session_state.vectorstore:
            with st.spinner("Initializing vector database..."):
                vectorstore, index_name = setup_pinecone(pinecone_key)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.index_name = index_name
                    st.session_state.pinecone_key = pinecone_key
                    st.success("✅ Vector database initialized!")
                else:
                    st.error("❌ Failed to initialize vector database")
                    return
        
        # File upload section
        st.subheader("📎 Upload Documents")
        
        # Upload mode selection
        upload_mode = st.radio(
            "Upload mode:",
            ["Multiple files", "Single file (legacy)"],
            horizontal=True,
            help="Choose upload mode - multiple files for batch processing or single file for compatibility"
        )
        
        if upload_mode == "Multiple files":
            uploaded_files = st.file_uploader(
                "Choose PDF files", 
                type="pdf",
                accept_multiple_files=True,
                help="Upload one or more PDF documents to analyze"
            )
        else:
            # Legacy single file mode
            uploaded_file = st.file_uploader(
                "Choose a PDF file", 
                type="pdf",
                help="Upload a PDF document to analyze (legacy mode)"
            )
            uploaded_files = [uploaded_file] if uploaded_file else None
        
        if uploaded_files:
            st.write(f"**Selected files:** {len(uploaded_files)}")
            
            # File validation
            MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
            valid_files = []
            invalid_files = []
            
            for i, file in enumerate(uploaded_files):
                file_size_mb = file.size / (1024 * 1024)
                if file.size > MAX_FILE_SIZE:
                    invalid_files.append(f"{file.name} ({file_size_mb:.1f}MB - too large)")
                else:
                    valid_files.append(file)
                    st.write(f"{i+1}. ✅ {file.name} ({file_size_mb:.1f}MB)")
            
            if invalid_files:
                st.error("⚠️ Some files are too large (max 50MB):")
                for invalid_file in invalid_files:
                    st.write(f"❌ {invalid_file}")
                uploaded_files = valid_files
        
        col1, col2 = st.columns([1, 4])
        with col1:
            process_btn = st.button("🚀 Process Documents", disabled=not uploaded_files)
        
        if process_btn and uploaded_files:
            if upload_mode == "Single file (legacy)" and len(uploaded_files) == 1:
                # Legacy single file processing
                with st.spinner("Processing PDF document..."):
                    success, result = process_pdf(uploaded_files[0], st.session_state.vectorstore)
                
                if success:
                    st.success(f"✅ Document processed successfully! Created {result} text chunks.")
                    st.session_state.documents_processed = True
                else:
                    st.error(f"❌ Error processing document: {result}")
            else:
                # Multiple files processing
                results = process_multiple_pdfs(uploaded_files, st.session_state.vectorstore)
                
                # Display results
                successful_files = [r for r in results if r["status"] == "success"]
                failed_files = [r for r in results if r["status"] == "error"]
                
                if successful_files:
                    total_chunks = sum(r["chunks"] for r in successful_files)
                    total_time = sum(r["processing_time"] for r in successful_files)
                    st.success(f"✅ {len(successful_files)} documents processed successfully! Created {total_chunks} text chunks in {total_time:.1f}s.")
                    st.session_state.documents_processed = True
                    
                    # Show successful files
                    with st.expander("📋 Successfully processed files"):
                        for result in successful_files:
                            st.write(f"✅ {result['file']} - {result['chunks']} chunks ({result['processing_time']:.1f}s)")
                
                if failed_files:
                    st.error(f"❌ {len(failed_files)} documents failed to process")
                    # Show failed files
                    with st.expander("⚠️ Failed files"):
                        for result in failed_files:
                            st.write(f"❌ {result['file']} - {result['error']}")
        
        # Question answering section
        if st.session_state.documents_processed:
            st.subheader("❓ Ask Questions")
            
            question = st.text_input(
                "Enter your question about the document:",
                placeholder="What is this document about?"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_btn = st.button("🔍 Get Answer", disabled=not question)
            
            if ask_btn and question:
                with st.spinner("Searching for answer..."):
                    answer = ask_question(question, st.session_state.vectorstore)
                
                st.subheader("💬 Answer:")
                st.write(answer)
                
                # Add to chat history
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append({"question": question, "answer": answer})
        
        # Chat history
        if st.session_state.get('chat_history'):
            with st.expander("📜 Chat History"):
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                    st.markdown(f"**Q{len(st.session_state.chat_history)-i}:** {chat['question']}")
                    st.markdown(f"**A:** {chat['answer']}")
                    st.markdown("---")
    
    else:
        st.warning("🔐 Please provide API keys to start using the application.")
        
        with st.expander("📋 Setup Instructions"):
            st.markdown("""
            ### Getting Started
            
            1. **Get API Keys:**
               - OpenAI: Visit [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
               - Pinecone: Visit [app.pinecone.io](https://app.pinecone.io/)
            
            2. **For Streamlit Cloud Deployment:**
               - Add keys to your app's secrets in the Streamlit Cloud dashboard
               - Format: `OPENAI_API_KEY = "your-key-here"`
            
            3. **Usage:**
               - Upload a PDF document
               - Click "Process Document"
               - Ask questions about the content
            """)

if __name__ == "__main__":
    main()