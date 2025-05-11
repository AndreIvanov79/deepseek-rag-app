import os
import shutil
import zipfile
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.settings import Settings  # This replaces ServiceContext
from llama_index.core.schema import Document
from llama_index.core.node_parser import CodeSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate

# from llama_index import ServiceContext, VectorStoreIndex, StorageContext, load_index_from_storage
# from llama_index import CodeSplitter
# from llama_index.schema import Document
# from llama_index_embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms import HuggingFaceLLM
# from llama_index.prompts import PromptTemplate

class RAGManager:
    def __init__(self, 
                 model_loader, 
                 index_store_dir: str = "./data/indices",
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG Manager for code context handling.
        
        Args:
            model_loader: Instance of ModelLoader class
            index_store_dir: Directory to store vector indices
            embed_model: Embedding model to use
        """
        self.model_loader = model_loader
        self.index_store_dir = index_store_dir
        self.embed_model = embed_model
        self.embedding_model = HuggingFaceEmbedding(model_name=embed_model)
        self.projects = {}  # Map of project_id to project name
        
        # Create index store directory
        os.makedirs(self.index_store_dir, exist_ok=True)
        
        # Load existing projects information
        self._load_projects_info()
    
    def _load_projects_info(self):
        """Load information about existing project indices."""
        self.projects = {}
        for item in os.listdir(self.index_store_dir):
            project_dir = os.path.join(self.index_store_dir, item)
            if os.path.isdir(project_dir):
                project_info_path = os.path.join(project_dir, "project_info.txt")
                if os.path.exists(project_info_path):
                    with open(project_info_path, 'r') as f:
                        project_name = f.read().strip()
                        self.projects[item] = project_name
    
    def _extract_zip(self, file_path: str) -> str:
        """
        Extract a file to a temporary directory. For zip files, extracts the contents.
        For other files, copies the file to the directory.

        Args:
            file_path: Path to the file

        Returns:
            Path to the extracted directory
        """
        temp_dir = tempfile.mkdtemp()

        # Check if file is a zip file
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        else:
            # For non-zip files, just copy the file to the temp directory
            file_name = os.path.basename(file_path)
            shutil.copy2(file_path, os.path.join(temp_dir, file_name))

        return temp_dir
    
    def _get_code_documents(self, project_dir: str, ignored_extensions: List[str] = None) -> List[Document]:
        """
        Get code documents from a project directory.

        Args:
            project_dir: Path to project directory
            ignored_extensions: File extensions to ignore

        Returns:
            List of Document objects
        """
        if ignored_extensions is None:
            ignored_extensions = [
                '.pyc', '.git', '.idea', '.vscode', '.DS_Store', '__pycache__',
                '.exe', '.dll', '.so', '.dylib', '.class', '.jar'
            ]

        binary_extensions = [
            '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.pdf', '.zip',
            '.tar', '.gz', '.rar', '.7z', '.bin', '.dat', '.db', '.sqlite'
        ]

        documents = []

        for root, _, files in os.walk(project_dir):
            for file in files:
                # Skip ignored folders/extensions
                if any(ext in root for ext in ignored_extensions):
                    continue

                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                relative_path = os.path.relpath(file_path, project_dir)

                # Handle different file types
                try:
                    # Skip binary files but include their metadata
                    if file_ext in binary_extensions:
                        document = Document(
                            text=f"Binary file: {relative_path}",
                            metadata={
                                'filename': relative_path,
                                'file_path': relative_path,
                                'file_type': file_ext,
                                'file_size': os.path.getsize(file_path),
                                'is_binary': True
                            }
                        )
                    else:
                        # Try to read as text file
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        document = Document(
                            text=content,
                            metadata={
                                'filename': relative_path,
                                'file_path': relative_path,
                                'file_type': file_ext,
                                'file_size': os.path.getsize(file_path),
                                'is_binary': False
                            }
                        )

                    documents.append(document)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

        return documents
    
    def create_index_from_zip(self, project_name: str, file_path: str) -> Tuple[str, int]:
        """
        Create a search index from a file or zipped project.

        Args:
            project_name: Name of the project
            file_path: Path to the file (zip or any other file)

        Returns:
            Tuple containing the project ID and number of documents indexed
        """
        # Extract to temporary directory
        extract_dir = self._extract_zip(file_path)

        try:
            # Get code documents
            documents = self._get_code_documents(extract_dir)

            if not documents:
                raise ValueError("No valid documents found in the file.")
            
            # Create project ID and directory
            project_id = f"project_{len(self.projects) + 1}"
            project_store_dir = os.path.join(self.index_store_dir, project_id)
            os.makedirs(project_store_dir, exist_ok=True)
            
            # Initialize code splitter for chunking code files
            splitter = CodeSplitter(
                language="python",
                chunk_lines=40,
                chunk_lines_overlap=15,
                max_chars=1500
            )
            
            # Create service context
            
            service_context = Settings.from_defaults(
                llm=None,
                embed_model=self.embedding_model,
                node_parser=splitter
            )
            
            # Create and persist index
            index = VectorStoreIndex.from_documents(
                documents,
                service_context=service_context
            )
            
            index.storage_context.persist(persist_dir=project_store_dir)
            
            # Save project info
            with open(os.path.join(project_store_dir, "project_info.txt"), 'w') as f:
                f.write(project_name)
            
            # Update projects dictionary
            self.projects[project_id] = project_name
            
            return project_id, len(documents)
        
        finally:
            # Clean up temporary directory
            shutil.rmtree(extract_dir, ignore_errors=True)
    
    def get_projects(self) -> Dict[str, str]:
        """
        Get all available projects.
        
        Returns:
            Dictionary mapping project IDs to project names
        """
        return self.projects
    
    def query_project(self, 
                     project_id: str, 
                     query: str, 
                     max_tokens: int = 4096,
                     similarity_top_k: int = 5) -> Dict[str, Any]:
        """
        Query a project index with a question.
        
        Args:
            project_id: ID of the project
            query: Query string
            max_tokens: Maximum tokens for response
            similarity_top_k: Number of top similar chunks to retrieve
            
        Returns:
            Dictionary containing the response and relevant context
        """
        if project_id not in self.projects:
            raise ValueError(f"Project ID {project_id} not found")
        
        project_store_dir = os.path.join(self.index_store_dir, project_id)
        
        # Load index from storage
        storage_context = StorageContext.from_defaults(persist_dir=project_store_dir)
        index = load_index_from_storage(storage_context)
        
        # Get model and tokenizer
        model, tokenizer = self.model_loader.get_model_and_tokenizer()
        
        # Create DeepSeek LLM wrapper
        llm = HuggingFaceLLM(
            model=model,
            tokenizer=tokenizer,
            context_window=max_tokens,
            max_new_tokens=max_tokens,
            model_kwargs={"temperature": 0.1}
        )
        
        # Create query engine with deepseek model
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=similarity_top_k
        )
        
        # Query the index
        response = query_engine.query(query)
        
        # Get source nodes for context
        context = []
        for source_node in response.source_nodes:
            context.append({
                "text": source_node.node.text[:500] + "..." if len(source_node.node.text) > 500 else source_node.node.text,
                "metadata": source_node.node.metadata,
                "score": round(source_node.score, 4) if hasattr(source_node, 'score') else None
            })
        
        return {
            "response": response.response,
            "context": context
        }
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project index.
        
        Args:
            project_id: ID of the project
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if project_id not in self.projects:
            return False
        
        project_store_dir = os.path.join(self.index_store_dir, project_id)
        
        # Delete project directory
        shutil.rmtree(project_store_dir, ignore_errors=True)
        
        # Remove from projects dictionary
        del self.projects[project_id]
        
        return True