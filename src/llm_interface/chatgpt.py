from openai import OpenAI
from typing import List, Dict, Any


class TextManager:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.conversation_history: List[Dict[str, str]] = []

    def text_to_text(self, system_prompt: str, user_prompt: str) -> str:
        """Simple text-to-text completion"""
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return completion.choices[0].message.content
