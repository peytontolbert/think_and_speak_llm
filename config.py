from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from .env file"""
    try:
        load_dotenv()
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
            
        return {
            "openai_api_key": openai_key
        }
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise 