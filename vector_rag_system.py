"""
Vector RAG System for Customer Support
Uses Qdrant with in-memory ephemeral storage for FAQ and Knowledge Base vector stores
"""

import os
import tiktoken
from typing import List, Dict, Any
from datetime import datetime

from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI as LangChainOpenAI
from qdrant_client import QdrantClient

from dotenv import load_dotenv

load_dotenv()

# LangSmith integration
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    # Fallback decorator if LangSmith not available
    def traceable(name=None):
        def decorator(func):
            return func
        return decorator
    LANGSMITH_AVAILABLE = False

class VectorRAGSystem:
    """Production RAG system with FAQ and KB vector stores"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required")
        
        # Initialize LangSmith tracing if available
        self._setup_langsmith()
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize embedding model
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key
        )
        
        # Initialize LLM for reranking
        self.llm = LangChainOpenAI(
            model="gpt-4o-mini",
            openai_api_key=self.openai_api_key,
            temperature=0
        )
        
        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        
        # Setup text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Restored to reasonable size for meaningful chunks
            chunk_overlap=50,  # Proportional overlap
            length_function=self._token_len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )
        
        # Initialize vector stores
        self.faq_vectorstore = None
        self.kb_vectorstore = None
        self.faq_retriever = None
        self.kb_retriever = None
        
        # Initialize vector stores (in-memory for ephemeral, stateless operation)
        self._initialize_vector_stores()
    
    def _setup_langsmith(self):
        """Setup LangSmith tracing if credentials are available"""
        if LANGSMITH_AVAILABLE:
            langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
            langsmith_project = os.getenv('LANGSMITH_PROJECT', 'MCP-A2A-Agents')
            
            if langsmith_api_key:
                os.environ['LANGSMITH_TRACING'] = 'true'
                os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
                os.environ['LANGSMITH_API_KEY'] = langsmith_api_key
                os.environ['LANGSMITH_PROJECT'] = langsmith_project
                print(f"✅ LangSmith tracing enabled for project: {langsmith_project}")
            else:
                print("⚠️  LangSmith API key not found, tracing disabled")
        else:
            print("⚠️  LangSmith not installed, tracing disabled")
    
    def _token_len(self, text: str) -> int:
        """Count tokens in text"""
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def _initialize_vector_stores(self):
        """Initialize ephemeral in-memory FAQ and KB vector stores"""
        try:
            # Create shared in-memory client
            self.qdrant_client = QdrantClient(location=":memory:")
            
            # Initialize empty vector stores - they will be created when first documents are added
            self.faq_vectorstore = None
            self.kb_vectorstore = None
            self.faq_retriever = None
            self.kb_retriever = None
            
            print("✅ In-memory vector stores initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing vector stores: {e}")
            raise
    
    def _setup_retrievers(self):
        """Setup retrievers with contextual compression (reranking)"""
        # Only setup retrievers if vector stores exist
        if self.faq_vectorstore:
            base_faq_retriever = self.faq_vectorstore.as_retriever(
                search_kwargs={"k": 10}  # Get more candidates for reranking
            )
            compressor = LLMChainExtractor.from_llm(self.llm)
            self.faq_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_faq_retriever
            )
        
        if self.kb_vectorstore:
            base_kb_retriever = self.kb_vectorstore.as_retriever(
                search_kwargs={"k": 10}  # Get more candidates for reranking
            )
            compressor = LLMChainExtractor.from_llm(self.llm)
            self.kb_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_kb_retriever
            )
    
    def create_faq_vectorstore(self, faq_content: str) -> bool:
        """Create/update FAQ vector store from FAQ content"""
        try:
            # Parse FAQ content into Q&A pairs
            faq_documents = self._parse_faq_content(faq_content)
            
            if not faq_documents:
                print("⚠️  No FAQ content to process")
                return False
            
            # Create new FAQ vector store with documents (this properly initializes the collection)
            self.faq_vectorstore = QdrantVectorStore.from_documents(
                documents=faq_documents,
                embedding=self.embedding_model,
                collection_name="faq_collection",
                url=":memory:",  # Use url parameter for in-memory
            )
            
            # Recreate retriever
            self._setup_retrievers()
            
            print(f"✅ FAQ vector store created with {len(faq_documents)} Q&A pairs")
            return True
            
        except Exception as e:
            print(f"❌ Error creating FAQ vector store: {e}")
            return False
    
    def create_kb_vectorstore(self, pdf_path: str) -> bool:
        """Create KB vector store from PDF file"""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Load PDF
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            print(f"📄 Loaded {len(documents)} pages from PDF: {pdf_path}")
            
            # Split documents into chunks
            split_chunks = self.text_splitter.split_documents(documents)
            print(f"📄 Created {len(split_chunks)} knowledge chunks")
            
            # Create new KB vector store with documents (this properly initializes the collection)
            self.kb_vectorstore = QdrantVectorStore.from_documents(
                documents=split_chunks,
                embedding=self.embedding_model,
                collection_name="kb_collection",
                url=":memory:",  # Use url parameter for in-memory
            )
            
            # Recreate retriever
            self._setup_retrievers()
            
            print(f"✅ KB vector store created with {len(split_chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error creating KB vector store: {e}")
            return False
    
    def _parse_faq_content(self, faq_content: str) -> List[Document]:
        """Parse FAQ markdown content into Q&A document pairs"""
        documents = []
        
        try:
            lines = faq_content.split('\n')
            current_question = ""
            current_answer = ""
            
            for line in lines:
                line = line.strip()
                
                # Question (starts with Q: or ##)
                if line.startswith('Q:') or line.startswith('## '):
                    # Save previous Q&A pair
                    if current_question and current_answer:
                        doc = Document(
                            page_content=f"Question: {current_question}\nAnswer: {current_answer}",
                            metadata={
                                "type": "faq",
                                "question": current_question,
                                "answer": current_answer,
                                "created_at": datetime.now().isoformat()
                            }
                        )
                        documents.append(doc)
                    
                    # Start new question
                    current_question = line.replace('Q:', '').replace('## ', '').strip()
                    current_answer = ""
                
                # Answer (starts with A: or follows question)
                elif line.startswith('A:'):
                    current_answer = line.replace('A:', '').strip()
                elif current_question and line and not line.startswith('#'):
                    # Continue answer
                    if current_answer:
                        current_answer += " " + line
                    else:
                        current_answer = line
            
            # Save last Q&A pair
            if current_question and current_answer:
                doc = Document(
                    page_content=f"Question: {current_question}\nAnswer: {current_answer}",
                    metadata={
                        "type": "faq",
                        "question": current_question,
                        "answer": current_answer,
                        "created_at": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ Error parsing FAQ content: {e}")
            return []
    
    @traceable(name="search_faq")
    async def search_faq(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Search FAQ vector store with similarity search and reranking"""
        try:
            if not self.faq_vectorstore or not self.faq_retriever:
                return {"found": False, "error": "FAQ vector store not initialized"}
            
            # Retrieve relevant FAQ documents
            docs = self.faq_retriever.get_relevant_documents(query)
            
            if not docs:
                return {"found": False, "confidence": 0.0, "source": "FAQ"}
            
            # Get best match
            best_doc = docs[0]
            answer = best_doc.metadata.get("answer", best_doc.page_content)
            
            # Calculate confidence based on similarity (simplified)
            confidence = min(0.9, 0.6 + len(docs) * 0.1)  # Higher confidence with more matches
            
            return {
                "found": True,
                "solution": answer,
                "confidence": confidence,
                "source": "FAQ",
                "question": best_doc.metadata.get("question", ""),
                "num_results": len(docs)
            }
            
        except Exception as e:
            print(f"❌ Error searching FAQ: {e}")
            return {"found": False, "error": str(e), "confidence": 0.0}
    
    @traceable(name="search_knowledge_base")
    async def search_knowledge_base(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search KB vector store with similarity search and reranking"""
        try:
            if not self.kb_vectorstore or not self.kb_retriever:
                return {"found": False, "error": "KB vector store not initialized"}
            
            # Retrieve relevant KB documents
            docs = self.kb_retriever.get_relevant_documents(query)
            
            if not docs:
                return {"found": False, "confidence": 0.0, "source": "Knowledge Base"}
            
            # Combine top documents into solution
            combined_content = ""
            for i, doc in enumerate(docs[:top_k]):
                combined_content += f"[Source {i+1}] {doc.page_content}\n\n"
            
            # Generate solution using LLM
            solution = await self._generate_solution_from_context(query, combined_content)
            
            # Calculate confidence
            confidence = min(0.8, 0.4 + len(docs) * 0.08)
            
            return {
                "found": True,
                "solution": solution,
                "confidence": confidence,
                "source": "Knowledge Base",
                "num_results": len(docs),
                "context_used": len(docs[:top_k])
            }
            
        except Exception as e:
            print(f"❌ Error searching KB: {e}")
            return {"found": False, "error": str(e), "confidence": 0.0}
    
    @traceable(name="generate_solution_from_context")
    async def _generate_solution_from_context(self, query: str, context: str) -> str:
        """Generate solution from retrieved context using LLM"""
        try:
            prompt = f"""Based on the following knowledge base context, provide a clear and helpful answer to the user's question.

User Question: {query}

Knowledge Base Context:
{context}

Instructions:
- Provide a direct, actionable answer
- Use information from the context provided
- If the context doesn't fully answer the question, say so
- Keep the response concise but complete
- Use a helpful, professional tone

Answer:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error generating solution: {e}")
            return f"Based on our knowledge base: {context[:300]}..."
    
    def update_faq_from_doc_change(self, updated_faq_content: str) -> bool:
        """Update FAQ vector store when FAQ document changes"""
        print("🔄 FAQ document updated, refreshing vector store...")
        return self.create_faq_vectorstore(updated_faq_content)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of in-memory vector stores"""
        try:
            faq_count = 0
            kb_count = 0
            faq_status = "empty"
            kb_status = "empty"
            
            # Check FAQ vector store
            if self.faq_vectorstore is not None:
                try:
                    # For in-memory stores, check if collections exist and have data
                    faq_info = self.faq_vectorstore.client.get_collection("faq_collection")
                    faq_count = faq_info.points_count if hasattr(faq_info, 'points_count') else 0
                    faq_status = "active" if faq_count > 0 else "initialized"
                except:
                    faq_status = "not_initialized"
            
            # Check KB vector store
            if self.kb_vectorstore is not None:
                try:
                    kb_info = self.kb_vectorstore.client.get_collection("kb_collection")
                    kb_count = kb_info.points_count if hasattr(kb_info, 'points_count') else 0
                    kb_status = "active" if kb_count > 0 else "initialized"
                except:
                    kb_status = "not_initialized"
            
            return {
                "storage_type": "in-memory (ephemeral)",
                "faq_vectorstore": {
                    "status": faq_status,
                    "document_count": faq_count,
                    "description": "FAQ Q&A pairs from Drive documents"
                },
                "kb_vectorstore": {
                    "status": kb_status,
                    "document_count": kb_count,
                    "description": "Knowledge base from uploaded PDF"
                },
                "embedding_model": "text-embedding-3-small",
                "llm_model": "gpt-4o-mini",
                "session_info": "Vector stores reset on server restart (stateless)"
            }
            
        except Exception as e:
            return {"error": str(e), "storage_type": "in-memory (ephemeral)"}

# Global instance
rag_system = None

def get_rag_system() -> VectorRAGSystem:
    """Get global RAG system instance"""
    global rag_system
    if rag_system is None:
        rag_system = VectorRAGSystem()
    return rag_system
