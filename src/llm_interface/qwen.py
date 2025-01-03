from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

class QwenManager:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        """Initialize Qwen model and tokenizer"""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.conversation_history: List[Dict[str, str]] = []

    def text_to_text(self, system_prompt: str, user_prompt: str) -> str:
        """Simple text-to-text completion using Qwen"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Prepare inputs
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate response
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode response
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response