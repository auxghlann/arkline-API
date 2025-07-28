from groq import Groq
from dotenv import load_dotenv
from app.utils.util import extract_clean_json
from app.client.prompts.arkline_urgency_prompts import SYSTEM_BASE_PROMPT, BEHAVIOR_FEW_SHOT_MESSAGE
import os

load_dotenv()

class ArklineAI:

    CATEGORY_FUNCTION_DEFINITION = {
        "type": "function",
        "function": {
            "name": "classify_urgency",
            "description": "Classify urgency of intern request based on the subject and message content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urgency": {
                        "type": "string",
                        "enum": ["High", "Medium", "Low", "Others"]
                    }
                },
                "required": ["urgency"]
            }
        }
    }

    def __init__(self) -> None:
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
        )

    def get_response(self, subject: str, message: str):
        prompt = self.__prompt_build(subject, message)
        
        # Build messages with system prompt
        messages = [{"role": "system", "content": SYSTEM_BASE_PROMPT}]
        
        # Add all training examples
        messages.extend(BEHAVIOR_FEW_SHOT_MESSAGE)
        
        # Add the actual user request
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            tools=[ArklineAI.CATEGORY_FUNCTION_DEFINITION], # type: ignore
            tool_choice="auto"
        )
        if not response.choices:
            return None
        
        # First try to get function call result
        if response.choices[0].message.tool_calls:
            return extract_clean_json(response.choices[0].message.tool_calls[0].function.arguments)
        
        # Fallback: try to extract JSON from regular message content
        if response.choices[0].message.content:
            fallback_result = extract_clean_json(response.choices[0].message.content)
            if fallback_result and 'urgency' in fallback_result:
                return fallback_result
        
        # Last resort: default to "Others" for any unclassifiable content
        return {"urgency": "Others"}


    def __prompt_build(self, subject: str, message: str) -> str:
        return f"""
        Subject: {subject}
        Message: {message}
        """