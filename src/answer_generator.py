from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
import sys
from .chat_venice_api import ChatVeniceAPI

class AnswerGenerator:
    def __init__(self, text: str, prompt: str, model: str, api_key: str, api_base: str, embedding_model: str = "llama3", verbose: bool = False):
        self.text = text
        self.prompt = prompt
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.verbose = verbose
        self.llm = ChatVeniceAPI(model=self.model, api_key=self.api_key, base_url=self.api_base)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_store = None
        self._setup_vector_store()

    def _log_verbose(self, message: str):
        if self.verbose:
            print(f"[VERBOSE] {message}", file=sys.stderr)

    def _setup_vector_store(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(self.text)
        self.vector_store = InMemoryVectorStore.from_texts(chunks, self.embeddings)

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

        self._log_verbose(f"Sending answer generation prompt: {full_prompt}")
        response = self.llm.invoke([message])
        self._log_verbose(f"Received answer response: {response.content}")

        # Check if the response indicates failure
        if response.content.strip() == "FAIL":
            return None

        return response.content.strip()
