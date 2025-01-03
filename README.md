# AI Conversation Agent

A sophisticated AI conversation agent that simulates natural dialogue through continuous cognitive processing and contextual awareness. The agent utilizes Large Language Models (LLMs) with vector embeddings for dynamic knowledge management and contextual understanding.

## Features

- **Continuous Cognitive Processing**: The agent maintains an active "thinking" state even when not speaking
- **Natural Conversation Flow**: Uses `<speak>` and `<standby>` modes to simulate natural dialogue patterns
- **Dynamic Context Management**: Continuously updates and maintains conversation context using vector embeddings
- **Memory Management**: Stores and retrieves relevant information through vector-based knowledge storage
- **Real-time Processing**: Maintains an ongoing loop for processing and updating contextual information

## How It Works

1. **Main Loop**
   - Continuously monitors conversation
   - Updates context and knowledge base
   - Manages the agent's state between speaking and processing

2. **State Management**
   - `<standby>`: Active listening and processing state
   - `<speak>`: Engagement and response state

3. **Vector Embeddings**
   - Stores conversation context
   - Maintains knowledge base
   - Enables semantic search and retrieval

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key or compatible LLM API
- Vector database (e.g., FAISS, Pinecone)

### Installation
```bash
git clone https://github.com/peytontolbert/think_and_speak_llm
cd think_and_speak_llm
pip install -r requirements.txt
```


### Configuration

1. Set up your environment variables:
```bash
export OPENAI_API_KEY=your_api_key_here
```

2. Configure your vector database settings in `config.py`

### Usage
```python
from conversation_agent import ConversationAgent
agent = ConversationAgent()
agent.start_conversation()
```


## Architecture

- **Main Loop**: Manages the conversation flow and state transitions
- **Context Manager**: Handles vector embeddings and knowledge updates
- **State Handler**: Controls the agent's speaking and standby states
- **Memory Module**: Manages long-term and working memory through vector storage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenAI for LLM capabilities
- Vector embedding libraries and research
- Natural conversation processing research
