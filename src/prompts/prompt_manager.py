from typing import List, Dict
from src.llm_interface.chatgpt import TextManager
from src.vector.vector_store import VectorStore

class PromptManager:
    def __init__(self, text_manager: TextManager, vector_store: VectorStore):
        self.text_manager = text_manager
        self.vector_store = vector_store
        self.conversation_history: List[Dict[str, str]] = []
        
    def process_interim_thought(self, current_context: str) -> str:
        """Process thoughts while user is speaking"""
        system_prompt = """
        As an AI assistant, analyze the ongoing speech and form thoughts about:
        1. The main topics being discussed
        2. Potential questions or concerns being raised
        3. Key points to address in your response
        Do not generate speech - only internal thoughts to help prepare a response.
        
        Use these formats:
        - <think>your internal thought process</think> for cognitive processing
        - <standby> when processing or listening
        """
        return self.text_manager.text_to_text(system_prompt, current_context)

    def cognitive_process(self, context: str) -> str:
        """Enhanced cognitive processing"""
        system_prompt = """You are an intelligent AI agent with deep cognitive processing capabilities.
        Analyze the context and knowledge to maintain sophisticated understanding.
        Generate internal thoughts and insights that would be valuable for future context.
        
        Analyze for:
        1. Emotional undertones
        2. Conversation topics and progress
        3. Unaddressed points
        4. Timing and turn-taking
        
        Use these formats:
        - <think>your internal thought process</think> for cognitive processing
        - <standby> when processing or listening
        - <speak>your message</speak> for regular speech
        - <speak><uninterruptible>your message</uninterruptible></speak> for important points
        
        Consider conversation flow, timing, and social cues before speaking."""
        
        full_context = f"""Previous context: {' '.join(msg['content'] for msg in self.conversation_history[-5:] if msg)}
        
        Current context:
        {context}"""
        
        response = self.text_manager.text_to_text(system_prompt, full_context)
        
        # Store any new thoughts generated
        if "<think>" in response:
            thoughts = [t.split("</think>")[0].strip() for t in response.split("<think>")[1:]]
            for thought in thoughts:
                self.vector_store.add_text(thought)
        
        return response

    def conversational_response(self, user_input: str) -> str:
        """Generate conversational responses"""
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Get recent context
        recent_context = self.vector_store.get_recent_context()
        
        system_prompt = """You are a natural conversational AI agent.
        Use the context to inform your responses while maintaining natural conversation.
        
        Response guidelines:
        1. Keep responses concise and engaging
        2. Use natural conversational tone
        3. Consider context and timing
        
        Use these formats:
        - <think>your internal thought process</think> for cognitive processing
        - <standby> to indicate you're listening and processing
        - <speak>your message</speak> for regular speech
        - <speak><uninterruptible>your message</uninterruptible></speak> for important points
        
        Maintain natural conversation flow and timing."""
        
        full_context = f"""Recent context:
        {recent_context}
        
        Current input:
        {user_input}"""
        
        response = self.text_manager.text_to_text(system_prompt, full_context)
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Store any new thoughts
        if "<think>" in response:
            thoughts = [t.split("</think>")[0].strip() for t in response.split("<think>")[1:]]
            for thought in thoughts:
                self.vector_store.add_text(thought)
        
        return response 