from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
import sys
import os
import hashlib

class AnswerGenerator:
    def __init__(self, text: str, prompt: str, llm, embedding_model: str = "llama3", verbose: bool = False, vector_store_path: str = "vector_store"):
        self.text = text
        self.prompt = prompt
        self.llm = llm
        self.verbose = verbose
        self.vector_store_path = vector_store_path
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_store = None

    def _log_verbose(self, message: str):
        if self.verbose:
            print(f"[VERBOSE] {message}", file=sys.stderr)

    def _get_text_hash(self) -> str:
        """Generate hash of the text for duplicate detection"""
        return hashlib.sha256(self.text.encode('utf-8')).hexdigest()

    def _setup_vector_store(self, db):
        """Set up vector store, checking for duplicates first"""
        text_hash = self._get_text_hash()

        # Check if this text was already imported
        if db.is_text_imported(text_hash):
            self._log_verbose(f"Text hash {text_hash} already imported, loading existing vector store")
            if os.path.exists(self.vector_store_path):
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return
            else:
                self._log_verbose(f"Vector store path {self.vector_store_path} not found, recreating")
                # Mark as not imported so it gets recreated
                db.mark_text_not_imported(text_hash)

        # Create new vector store or add to existing one
        self._log_verbose(f"Processing text hash {text_hash}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(self.text)

        if chunks:
            # Check if vector store already exists
            if os.path.exists(self.vector_store_path):
                self._log_verbose(f"Adding to existing vector store at {self.vector_store_path}")
                # Load existing vector store
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                # Add new texts to existing store
                self.vector_store.add_texts(chunks)
            else:
                self._log_verbose(f"Creating new vector store at {self.vector_store_path}")
                # Create new vector store
                self.vector_store = FAISS.from_texts(chunks, self.embeddings)

            # Save the updated vector store
            self.vector_store.save_local(self.vector_store_path)
            # Mark as imported
            db.mark_text_imported(text_hash)
            self._log_verbose(f"Vector store updated and saved to {self.vector_store_path}")
        else:
            self._log_verbose("No chunks created from text")

    def load_or_create_vector_store(self, db):
        """Public method to initialize vector store with duplicate checking"""
        self._setup_vector_store(db)

    def load_existing_vector_store(self):
        """Load existing vector store without creating new one"""
        if os.path.exists(self.vector_store_path):
            self._log_verbose(f"Loading existing vector store from {self.vector_store_path}")
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self._log_verbose(f"Vector store path {self.vector_store_path} not found")

    def get_additional_context(self, question: str, k: int = 3) -> str:
        if not self.vector_store:
            return ""

        docs = self.vector_store.similarity_search(question, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    def generate_answer(self, question: str, primary_context: str) -> str:
        additional_context = self.get_additional_context(question)

        full_context = f"Primary Context:\n{primary_context}\n\nAdditional Context:\n{additional_context}"
        full_prompt = f"{self.prompt}\n\nContext:\n{full_context}\n\nQuestion: {question}"

        message = HumanMessage(content=full_prompt)

        try:
            self._log_verbose(f"Sending answer generation prompt: {full_prompt}")
            response = self.llm.invoke([message])
            self._log_verbose(f"Received answer response: {response.content}")
        except Exception as e:
            self._log_verbose(f"Answer generation failed after retries: {e}")
            return None

        # Check if the response indicates failure
        if response.content.strip() == "FAIL":
            return None

        return response.content.strip()
