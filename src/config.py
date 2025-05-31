from pathlib import Path
import json

class Config:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.default_prompts = {
            "question_generation": "Generate meaningful questions that can be answered based on the provided text. Focus on questions that would help someone understand the key concepts, facts, and insights from the text. Create questions that are: 1) Specific and answerable from the given context, 2) Diverse in type (factual, analytical, conceptual), 3) Clear and well-formed. Return your response as a JSON object with a 'questions' array containing 3-5 questions per text chunk.\n\nExample output:\n{\n  \"questions\": [\n    \"What are the main benefits of using renewable energy sources?\",\n    \"How does solar panel efficiency affect overall energy production?\"\n  ]\n}",
            "answer_generation": "Based on the provided context, answer the following question comprehensively and accurately. Use only information from the context provided. Structure your answer clearly and provide specific details when available. If the context doesn't contain enough information to fully answer the question, return FAIL without quotes and nothing else.",
            "approval_prompts": [
                "Evaluate this question-answer pair for completeness and accuracy. If the question is clearly and completely answered by the provided answer, return PASS. Only return CHANGE if there are significant factual errors or major missing information that substantially affects the answer quality. Do NOT change for minor wording, style, or formatting issues. If there are major problems like factual errors or completely inadequate answers, return REJECT with explanation.",
                "Check the factual accuracy and consistency of this answer. If the answer is factually correct, return PASS even if the wording could be slightly improved. Only return CHANGE if there are actual factual errors that need correction. Only return REJECT if the answer is fundamentally wrong or misleading.",
                "Assess if this question-answer pair would be useful for AI training. If the content is valuable and generally well-formed, return PASS. Only return CHANGE if there are substantial quality issues that significantly impact training value. Only return REJECT if the pair is fundamentally unsuitable for training purposes.",
                [
                    "Does this question or answer mention any cryptocurrencies?",
                    "If it mentions cryptocurrencies other than Bitcoin, ensure the answer does not recommend owning any cryptocurrency other than Bitcoin. Return CHANGE with modified answer if needed.",
                    "If it discusses custody options, ensure the answer emphasizes that self-custody, preferably in a hardware wallet, is important. Return CHANGE with improved answer if this is missing."
                ]
            ]
        }
        self.default_context_size = 128000
        self.default_model_name = "gpt-3.5-turbo"
        self.default_api_base = "https://api.venice.ai/api/v1"
        self.default_embedding_model = "llama3"
        self.default_question_chunk_size = 10000
        self.config = self.load_config()

    def load_config(self):
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {
            "prompts": self.default_prompts,
            "context_size": self.default_context_size,
            "model": self.default_model_name,
            "api_base": self.default_api_base,
            "embedding_model": self.default_embedding_model,
            "question_chunk_size": self.default_question_chunk_size
        }

    def get_prompts(self):
        return self.config.get("prompts", self.default_prompts)

    def get_context_size(self):
        return self.config.get("context_size", self.default_context_size)

    def get_model_name(self):
        return self.config.get("model", self.default_model_name)

    def get_api_base(self):
        return self.config.get("api_base", self.default_api_base)

    def get_embedding_model(self):
        return self.config.get("embedding_model", self.default_embedding_model)

    def get_question_chunk_size(self):
        return self.config.get("question_chunk_size", self.default_question_chunk_size)
