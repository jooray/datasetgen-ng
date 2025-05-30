from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
import sys

class Summarizer:
    def __init__(self, transcript: str, prompts: dict, context_size: int, llm, verbose: bool = False):
        self.transcript = transcript
        self.prompts = prompts
        self.context_size = context_size
        self.llm = llm
        self.verbose = verbose

    def split_transcript(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.context_size,
            chunk_overlap=0
        )
        return text_splitter.split_text(self.transcript)

    def generate_summary(self, text_chunk: str, prompt: str):
        message = HumanMessage(content=f"{prompt}\n\n{text_chunk}")
        try:
            response = self.llm.invoke([message])
            return response.content.strip()
        except Exception as e:
            if self.verbose:
                print(f"[VERBOSE] Summary generation failed after retries: {e}", file=sys.stderr)
            return "Summary generation failed after retries"

    def summarize(self):
        chunks = self.split_transcript()
        short_summary = self.generate_summary(chunks[0], self.prompts['short_summary'])

        if len(chunks) == 1:
            detailed_summary = self.generate_summary(chunks[0], self.prompts['detailed_summary'])
        else:
            chunk_summaries = []
            for chunk in chunks:
                summary = self.generate_summary(chunk, self.prompts['detailed_summary'])
                if summary != "Summary generation failed after retries":
                    chunk_summaries.append(summary)

            if chunk_summaries:
                detailed_summary = self.generate_summary("\n\n".join(chunk_summaries), self.prompts['merge_summary'])
            else:
                detailed_summary = "Detailed summary generation failed after retries"

        return short_summary, detailed_summary
