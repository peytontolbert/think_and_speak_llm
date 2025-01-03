import os
from typing import List, Dict
from llm_interface.chatgpt import TextManager
from vector.vector_store import VectorStore
from config import load_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBuilder:
    def __init__(self, api_key: str):
        self.text_manager = TextManager(api_key)
        self.vector_store = VectorStore()
        
    def generate_cognitive_knowledge(self, topic: str, depth: int = 3) -> None:
        """Generate and store deep knowledge about a topic"""
        system_prompt = """You are an AI knowledge generator. 
        Generate deep, insightful thoughts about the given topic.
        Each thought should be self-contained and provide valuable context.
        Focus on analytical, practical, and conceptual understanding."""
        
        for i in range(depth):
            prompt = f"""Topic: {topic}
            Depth Level: {i + 1}
            Generate 5 deep insights about this topic. 
            Each insight should build upon previous knowledge and reveal deeper understanding.
            Focus on aspects that would be valuable for an AI to know during conversations."""
            
            response = self.text_manager.text_to_text(system_prompt, prompt)
            thoughts = [t.strip() for t in response.split('\n') if t.strip()]
            
            for thought in thoughts:
                self.vector_store.add_text(thought)
                logger.info(f"Added knowledge: {thought[:100]}...")

def build_initial_knowledge():
    """Build initial knowledge store with various topics"""
    try:
        config = load_config()
        api_key = config["openai_api_key"]
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise
    
    builder = KnowledgeBuilder(api_key)
    
    # List of topics to generate knowledge about
    topics = [
        "human conversation patterns",
        "active listening techniques",
        "emotional intelligence",
        "social cues and timing",
        "conversation flow management",
        "context awareness in discussions",
        "natural language understanding",
        "cognitive processing patterns",
        "memory and context in conversations",
        "social interaction dynamics"
    ]
    
    for topic in topics:
        logger.info(f"Generating knowledge about: {topic}")
        builder.generate_cognitive_knowledge(topic)
        
if __name__ == "__main__":
    build_initial_knowledge() 