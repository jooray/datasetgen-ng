from typing import List, Literal, Optional, Union
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field
import sys
import json

class PassResponse(BaseModel):
    status: Literal["PASS"] = "PASS"

class RejectResponse(BaseModel):
    status: Literal["REJECT"] = "REJECT"
    message: str = Field(description="Explanation of why the question-answer pair was rejected")

class ChangeResponse(BaseModel):
    status: Literal["CHANGE"] = "CHANGE"
    question: str = Field(description="The improved/corrected question")
    answer: str = Field(description="The improved/corrected answer")
    justification: str = Field(description="Explanation of what was changed and why")

class ApprovalChecker:
    def __init__(self, approval_prompts: List[Union[str, List[str]]], llm, verbose: bool = False):
        self.approval_prompts = approval_prompts
        self.llm = llm
        self.verbose = verbose

    def _log_verbose(self, message: str):
        if self.verbose:
            print(f"[VERBOSE] {message}", file=sys.stderr)

    def _check_yes_no_prompt(self, prompt: str, question: str, answer: str) -> bool:
        """Check a YES/NO prompt and return True if YES, False if NO"""
        full_prompt = f"{prompt}\n\nRespond with only YES or NO.\n\nQuestion: {question}\nAnswer: {answer}"
        message = HumanMessage(content=full_prompt)

        try:
            self._log_verbose(f"Sending YES/NO prompt: {full_prompt}")
            response = self.llm.invoke([message])
            self._log_verbose(f"Received YES/NO response: {response.content}")
        except Exception as e:
            self._log_verbose(f"YES/NO check failed after retries: {e}")
            return False

        # Check if response contains YES (case insensitive)
        response_text = response.content.strip().upper()
        return "YES" in response_text

    def _check_single_prompt(self, prompt: str, question: str, answer: str) -> dict:
        """Check a single approval prompt and return the structured response"""
        format_instructions = """Return your response as a JSON object with one of these formats:

For approval:
{
  "status": "PASS"
}

For rejection:
{
  "status": "REJECT",
  "message": "Explanation of why the question-answer pair was rejected"
}

For changes:
{
  "status": "CHANGE",
  "question": "The improved/corrected question",
  "answer": "The improved/corrected answer",
  "justification": "Explanation of what was changed and why"
}"""

        full_prompt = f"{prompt}\n\n{format_instructions}\n\nQuestion: {question}\nAnswer: {answer}"
        message = HumanMessage(content=full_prompt)

        try:
            self._log_verbose(f"Sending approval prompt: {full_prompt}")
            response = self.llm.invoke([message])
            self._log_verbose(f"Received approval response: {response.content}")
        except Exception as e:
            self._log_verbose(f"Approval check failed after retries: {e}")
            return {"status": "REJECT", "message": f"API call failed after retries: {str(e)}"}

        try:
            # Clean up the response content by removing markdown code blocks
            cleaned_content = response.content.strip()
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.startswith('```'):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]

            # Additional cleaning for control characters and formatting issues
            cleaned_content = cleaned_content.strip()

            # More comprehensive control character removal (preserve newlines in JSON strings)
            import re

            # Replace smart quotes with regular quotes
            cleaned_content = cleaned_content.replace('"', '"').replace('"', '"')
            cleaned_content = cleaned_content.replace(''', "'").replace(''', "'")

            # Handle escaped characters in JSON strings properly
            # First, let's try to parse and handle the JSON more carefully
            try:
                # Try parsing with Python's built-in JSON decoder which handles escapes
                parsed_response = json.loads(cleaned_content)

                if parsed_response.get("status") in ["PASS", "REJECT", "CHANGE"]:
                    # Validate CHANGE response has at least one of question or answer
                    if parsed_response.get("status") == "CHANGE":
                        if "question" not in parsed_response and "answer" not in parsed_response:
                            self._log_verbose("CHANGE response missing both question and answer fields")
                            return {"status": "REJECT", "message": "CHANGE response must include either question or answer field"}
                    return parsed_response
                else:
                    self._log_verbose(f"Invalid status in response: {parsed_response.get('status')}")
                    return {"status": "REJECT", "message": "Invalid approval response format"}

            except json.JSONDecodeError as json_error:
                self._log_verbose(f"First JSON parse attempt failed: {json_error}")

                # Try more aggressive cleaning for malformed JSON
                # Remove control characters but be more careful about what we remove
                cleaned_content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned_content)

                # Try to fix common JSON issues
                # Replace unescaped newlines in strings with escaped ones
                # This is tricky - we need to be careful not to break actual JSON structure

                # Try a different approach: extract the JSON structure manually if needed
                try:
                    parsed_response = json.loads(cleaned_content)

                    if parsed_response.get("status") in ["PASS", "REJECT", "CHANGE"]:
                        if parsed_response.get("status") == "CHANGE":
                            if "question" not in parsed_response and "answer" not in parsed_response:
                                self._log_verbose("CHANGE response missing both question and answer fields")
                                return {"status": "REJECT", "message": "CHANGE response must include either question or answer field"}
                        return parsed_response
                    else:
                        self._log_verbose(f"Invalid status in response: {parsed_response.get('status')}")
                        return {"status": "REJECT", "message": "Invalid approval response format"}

                except json.JSONDecodeError as second_error:
                    self._log_verbose(f"Second JSON parse attempt failed: {second_error}")

                    # Final fallback: try to extract just the status and create a minimal response
                    content_upper = response.content.upper()
                    if '"STATUS": "PASS"' in content_upper or '"STATUS":"PASS"' in content_upper:
                        return {"status": "PASS"}
                    elif '"STATUS": "REJECT"' in content_upper or '"STATUS":"REJECT"' in content_upper:
                        return {"status": "REJECT", "message": "Could not parse structured response but detected REJECT"}
                    elif '"STATUS": "CHANGE"' in content_upper or '"STATUS":"CHANGE"' in content_upper:
                        # For CHANGE, we need to be more careful since we need the actual content
                        # Return a REJECT instead since we can't safely extract the changes
                        return {"status": "REJECT", "message": "CHANGE response detected but could not parse JSON content safely"}
                    else:
                        return {"status": "REJECT", "message": f"Failed to parse JSON response: {str(second_error)}"}

        except Exception as e:
            self._log_verbose(f"Error parsing approval response: {e}")
            self._log_verbose(f"Raw response content: {response.content}")
            return {"status": "REJECT", "message": f"Failed to parse approval response: {str(e)}"}

    def check_approval(self, question: str, answer: str) -> dict:
        for i, prompt_item in enumerate(self.approval_prompts):
            if isinstance(prompt_item, list):
                # Conditional prompt array
                if len(prompt_item) == 0:
                    continue

                # First prompt is YES/NO gate
                gate_prompt = prompt_item[0]
                self._log_verbose(f"Processing conditional prompt group {i+1}: checking gate condition")

                should_process = self._check_yes_no_prompt(gate_prompt, question, answer)

                if not should_process:
                    self._log_verbose(f"Gate condition returned NO, skipping remaining prompts in group {i+1}")
                    continue  # Skip to next prompt group

                self._log_verbose(f"Gate condition returned YES, processing {len(prompt_item)-1} conditional prompts")

                # Process remaining prompts in the array
                for j, conditional_prompt in enumerate(prompt_item[1:], 1):
                    self._log_verbose(f"Processing conditional prompt {i+1}.{j}")
                    result = self._check_single_prompt(conditional_prompt, question, answer)

                    if result["status"] != "PASS":
                        # Add prompt context to rejection reason
                        if result["status"] == "REJECT" and "message" in result:
                            result["message"] = f"Conditional prompt {i+1}.{j}: {result['message']}"
                        return result  # Return REJECT or CHANGE immediately

            else:
                # Regular single prompt
                self._log_verbose(f"Processing single prompt {i+1}")
                result = self._check_single_prompt(prompt_item, question, answer)

                if result["status"] != "PASS":
                    # Add prompt context to rejection reason
                    if result["status"] == "REJECT" and "message" in result:
                        result["message"] = f"Prompt {i+1}: {result['message']}"
                    return result  # Return REJECT or CHANGE immediately

        # If all prompts passed, return PASS
        return {"status": "PASS"}
