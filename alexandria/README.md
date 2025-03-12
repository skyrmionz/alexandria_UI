# Alexandria - LangGraph Agent Implementation

Alexandria is a knowledge management system powered by LangGraph, a framework for building agentic workflows with LLMs.

## Features

- **Flexible Agent Architecture**: Uses LangGraph to create a dynamic agent that can use tools without explicit sequential workflows
- **Multiple Tools**: Includes knowledge base search, web search, text summarization, and calculation tools
- **Conversation Memory**: Maintains conversation history across interactions
- **Multiple Agent Types**: Supports different agent personas (Wise and Scribe) for different tasks
- **Tool Prioritization**: Intelligently prioritizes tools based on the query type and agent mode
- **Document Management**: Provides specialized document handling capabilities in Scribe mode

## Agent Modes

### Wise Mode
The Wise agent is designed to answer questions using a prioritized approach:
1. First checks the knowledge base for relevant information
2. If no relevant information is found, searches the web
3. Handles follow-up questions by referencing conversation history
4. Uses summarization and calculation tools as needed

### Scribe Mode
The Scribe agent specializes in document management:
1. Helps users find, create, and revise documents in the knowledge base
2. Maintains document context across multiple conversation turns
3. Provides proper formatting for document creation and editing
4. Creates knowledge artifacts with consistent structure

## Architecture

The system uses LangGraph's StateGraph to create a flexible agent architecture:

1. **Entry Point**: Checks if the query is a simple greeting, and if so, responds directly without using the full agent
2. **Context Detection**: Identifies follow-up questions and maintains context across conversation turns
3. **Agent Node**: Uses the LLM to decide what to do next (respond directly or use a tool)
4. **Tools Node**: Executes the selected tool and returns the result to the agent
5. **Conditional Routing**: Uses LangGraph's conditional edges to route between nodes based on the agent's decisions

## Tools

The agent has access to the following tools:

- **Knowledge Base Search**: Searches the user's personal knowledge base using vector similarity
- **Web Search**: Searches the web using DuckDuckGo
- **Text Summarization**: Summarizes long pieces of text
- **Calculator**: Performs mathematical calculations
- **List Artifacts**: Lists all available knowledge artifacts (Scribe mode only)
- **Get Artifact**: Retrieves a specific knowledge artifact by name (Scribe mode only)

## Usage

### Web Interface

The system provides a web interface for interacting with the agent:

1. **Chat**: Ask questions and get answers from the agent
2. **File Management**: Upload files to the knowledge base
3. **Knowledge Artifacts**: Create and manage knowledge artifacts
4. **Agent Mode Selection**: Switch between Wise and Scribe modes based on your needs

### API

The system also provides an API for programmatic access:

```python
from chat_agent import get_agent_response, update_agent

# Switch to Wise mode for answering questions
update_agent("wise")

# Get a response from the agent
response = get_agent_response("What is the capital of France?")
print(response)

# Switch to Scribe mode for document management
update_agent("scribe")

# Create or edit a document
response = get_agent_response("Create a new marketing strategy document")
print(response)
```

## Installation

1. Clone the repository
2. Install the dependencies: `pip install -r requirements.txt`
3. Set up environment variables:
   - `GITHUB_TOKEN`: GitHub token for accessing the Meta-Llama model
   - `FLASK_SECRET_KEY`: Secret key for Flask sessions
4. Run the application: `python app.py`

## Configuration

The system can be configured using environment variables:

- `GITHUB_TOKEN`: GitHub token for accessing the Meta-Llama model
- `FLASK_SECRET_KEY`: Secret key for Flask sessions
- `QDRANT_HOST`: Host for the Qdrant vector database (default: localhost)
- `QDRANT_PORT`: Port for the Qdrant vector database (default: 6333)

## LangGraph vs. LangChain

This implementation uses LangGraph instead of LangChain for the agent architecture. The key differences are:

1. **Flexible Workflows**: LangGraph allows for more flexible workflows with conditional routing
2. **State Management**: LangGraph provides better state management with typed states
3. **Tool Usage**: LangGraph makes it easier to use tools without explicit sequential workflows
4. **Debugging**: LangGraph provides better debugging capabilities with tracing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 