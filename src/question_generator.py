from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import RootModel
import sys
import json
from .chat_venice_api import ChatVeniceAPI

class QuestionList(RootModel[List[str]]):
    root: List[str]

class QuestionGenerator:
    def __init__(self, text: str, prompt: str, model: str, api_key: str, api_base: str, chunk_size: int = 10000, verbose: bool = False):
        self.text = text
        self.prompt = prompt
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.llm = ChatVeniceAPI(model=self.model, api_key=self.api_key, base_url=self.api_base)

    def _log_verbose(self, message: str):
        if self.verbose:
            print(f"[VERBOSE] {message}", file=sys.stderr)

    def split_text(self) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=200
        )
        return text_splitter.split_text(self.text)

    def generate_questions_for_chunk(self, chunk: str) -> List[str]:
        format_instructions = "Return your response as a JSON array of strings containing 3-5 questions.\n\nExample output:\n[\n  \"What are the main benefits of using renewable energy sources?\",\n  \"How does solar panel efficiency affect overall energy production?\"\n]"
        full_prompt = f"{self.prompt}\n\n{format_instructions}\n\nText:\n{chunk}"

        message = HumanMessage(content=full_prompt)

        self._log_verbose(f"Sending prompt to model: {full_prompt}")
        response = self.llm.invoke([message])
        self._log_verbose(f"Received response: {response.content}")

        try:
            # Clean up the response content by removing markdown code blocks
            cleaned_content = response.content.strip()
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]  # Remove ```json
            if cleaned_content.startswith('```'):
                cleaned_content = cleaned_content[3:]   # Remove ```
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]  # Remove trailing ```

            # Try to parse as JSON array directly
            parsed_response = json.loads(cleaned_content.strip())
            if isinstance(parsed_response, list):
                return parsed_response
            elif isinstance(parsed_response, dict) and 'questions' in parsed_response:
                return parsed_response['questions']
            else:
                self._log_verbose(f"Unexpected response format: {type(parsed_response)}")
                return []
        except Exception as e:
            self._log_verbose(f"Error parsing response: {e}")
            self._log_verbose(f"Raw response content: {response.content}")
            return []

    def generate_all_questions(self) -> List[tuple]:
        chunks = self.split_text()
        all_questions = []

        for chunk in chunks:
            questions = self.generate_questions_for_chunk(chunk)
            for question in questions:
                all_questions.append((question, chunk))

        return all_questions
