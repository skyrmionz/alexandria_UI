import os
import re
import json
import math
import time
import structlog
import requests
from typing import List, Optional, TypedDict, Literal, Dict, Any, TypeVar, Generic, Tuple, Union, get_args, get_origin
from functools import lru_cache
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from enum import Enum

# LangChain & related imports
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManager

# Local imports (adjust if needed)
from tools.knowledge_base_tool import (
    search_knowledge_base,
    search_artifacts as kb_search_artifacts,
    search_documents as kb_search_documents,
)
from tools.web_search_tool import web_search_tool

from langgraph.graph import StateGraph, END, START
from langgraph.types import Command

# -------------------------------------------------------------------------
# OpenRouterWrapper - Wrapper for OpenRouter API with DeepSeek-R1
# -------------------------------------------------------------------------
class OpenRouterWrapper(BaseChatModel):
    """Wrapper around OpenRouter API for DeepSeek-R1 model."""

    def __init__(self, api_key=None, model="deepseek/deepseek-r1:free", temperature=0.1):
        """Initialize the OpenRouterWrapper."""
        super().__init__()
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self._model = model
        self._temperature = temperature
        self._max_tokens = 8192  # Maximize token output for all responses
        self._endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        """Generate from OpenRouter with stops to prevent unnecessary token use."""
        logger.debug("Sending message to OpenRouter (DeepSeek-R1)", num_messages=len(messages))

        # Convert LangChain messages to OpenRouter messages (similar to OpenAI format)
        or_messages = []
        for message in messages:
            if isinstance(message, tuple) and len(message) == 2:
                role, content = message
                if role == "system":
                    or_messages.append({"role": "system", "content": content})
                elif role == "human":
                    or_messages.append({"role": "user", "content": content})
                elif role == "ai":
                    or_messages.append({"role": "assistant", "content": content})
            else:
                content = message.content
                if isinstance(message, SystemMessage):
                    or_messages.append({"role": "system", "content": content})
                elif isinstance(message, HumanMessage):
                    or_messages.append({"role": "user", "content": content})
                elif isinstance(message, AIMessage):
                    or_messages.append({"role": "assistant", "content": content})

        # Prepare the request payload
        payload = {
            "model": self._model,
            "messages": or_messages,
            "max_tokens": self._max_tokens,  # Default to high max_tokens
        }

        # Add optional parameters if provided
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs.get("temperature")
        else:
            payload["temperature"] = self._temperature

        if kwargs.get("max_tokens") is not None:
            payload["max_tokens"] = kwargs.get("max_tokens")

        # Add stop sequences if provided
        if stop:
            payload["stop"] = stop

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://alexandria-app.com",  # Replace with your domain
            "X-Title": "Alexandria AI Assistant",          # Replace with your app name
        }

        logger.debug(
            "OpenRouter API Request",
            url=self._endpoint,
            model=self._model,
            payload_size=len(json.dumps(payload)),
        )

        # Make the API call
        try:
            response = requests.post(self._endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            logger.debug("Received response from OpenRouter (DeepSeek-R1)")

            # Extract the content
            try:
                message_content = result["choices"][0]["message"]["content"]
                message = AIMessage(content=message_content)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            except (KeyError, IndexError) as e:
                logger.error(
                    "Error parsing API response", error=str(e), response=result
                )
                raise ValueError(f"Unexpected API response format: {result}")

        except requests.exceptions.HTTPError as e:
            logger.exception(f"HTTP Error calling OpenRouter API: {e}")
            if "429" in str(e):
                raise ValueError(
                    "The OpenRouter API is currently rate limited. Please try again later."
                )
            elif "401" in str(e) or "403" in str(e):
                raise ValueError(
                    "Authentication failed with OpenRouter API. Please check your API key."
                )
            else:
                raise ValueError(f"OpenRouter API Error: {str(e)}. Check configuration.")
        except Exception as e:
            logger.exception("Error calling OpenRouter API", error=str(e))
            raise

    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "openrouter-deepseek"

    @property
    def _identifying_params(self):
        """Return identifying parameters."""
        return {"model": self._model}

    def invoke(self, messages, **kwargs):
        """Generate a chat response."""
        processed_messages = []
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, tuple) and len(message) == 2:
                    role, content = message
                    if role == "system":
                        processed_messages.append(SystemMessage(content=content))
                    elif role == "human":
                        processed_messages.append(HumanMessage(content=content))
                    elif role == "ai":
                        processed_messages.append(AIMessage(content=content))
                else:
                    processed_messages.append(message)
        else:
            processed_messages = messages

        chat_result = self._generate(processed_messages, **kwargs)
        if not chat_result.generations:
            return AIMessage(content="I'm sorry, I wasn't able to generate a response.")
        return chat_result.generations[0].message


# -------------------------------------------------------------------------
# Set up structured logging
# -------------------------------------------------------------------------
logger = structlog.get_logger()

# -------------------------------------------------------------------------
# DEBUG: Print environment info to confirm environment variables are loaded.
# -------------------------------------------------------------------------
logger.info("Loading environment variables")
load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
azure_endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT", "https://models.inference.ai.azure.com")
logger.info(
    "Environment setup complete",
    github_token_present=bool(github_token),
    openrouter_api_key_present=bool(openrouter_api_key),
    azure_endpoint=azure_endpoint,
)

# -------------------------------------------------------------------------
# AgentState typed dict for the LangGraph state.
# -------------------------------------------------------------------------
class AgentState(TypedDict):
    """The state of our agent."""

    messages: List[BaseMessage]
    tools: dict
    tool_calls: list
    last_error: Optional[str]
    current_agent: str
    document_query: Optional[str]
    revision_query: Optional[str]
    revised_document: Optional[str]


# -------------------------------------------------------------------------
# Default prompts
# -------------------------------------------------------------------------
DEFAULT_WISE_PROMPT = """You are Alexandria, a highly capable AI assistant powered by DeepSeek-R1.
Your goal is to provide helpful, accurate, and contextually appropriate responses to users.

You have complete autonomy in deciding how to respond to user queries:

1. For greetings or casual conversation, you can respond naturally without using tools.
2. For simple factual questions (like "What is the capital of China?"), answer immediately from your knowledge without using tools.
3. For information requests that require up-to-date or specialized knowledge, use the appropriate tools.
4. For complex questions, you may need to use multiple tools to provide a comprehensive answer.
5. For document revision requests, delegate to the Scribe Agent who specializes in document modification.

IMPORTANT: Be efficient. Don't use tools when you already know the answer. Prioritize direct responses for simple questions.

WHEN TO DELEGATE TO THE SCRIBE AGENT:
- When a user asks to modify, edit, update, or revise a document
- When a user wants to insert, delete, or change content in a document
- When a user wants to format or restructure a document

To delegate a task to the Scribe Agent, include this in your response:

<delegate_document_revision>
document_query: "identification of the document to modify"
revision_query: "specific changes to make to the document"
</delegate_document_revision>

The Scribe Agent will process your request and return the revised document, which you can then present to the user.

WHEN TO CREATE KNOWLEDGE ARTIFACTS:
- When a user asks you to save information as an article, document, or artifact
- When you've revised a document and the user wants to save the changes as a new artifact
- When you want to preserve important information from a conversation in a structured format
- When creating reference materials from your research

FORMAT YOUR RESPONSES USING HTML:
- Use <h2> for section titles
- Use <h3> for subsection headings
- Use <ul> and <li> for bullet point lists
- Use <ol> and <li> for numbered lists
- Use <b> for emphasis on important points
- Use <i> for definitions or specialized terms
- Use <blockquote> for quoted content
- Use <hr> for separating major sections
- Use <em> to highlight discrepancies or warnings
DO NOT use color styling tags (like style="color:red").

You have access to the following tools:

- search_knowledge_base
- search_documents
- search_artifacts
- search_web
- calculate
- summarize_text
- list_artifacts
- get_artifact
- create_artifact

To use a tool, format your response like this:

<tool>
{
  "tool_name": "search_knowledge_base",
  "parameters": {
    "query": "example search query"
  }
}
</tool>

Remember that you are an autonomous agent - you decide when to use tools, when to respond directly, and when to delegate to the Scribe Agent. Be conversational and helpful, and use your best judgment to determine the appropriate response for each user query.
"""

DEFAULT_SCRIBE_PROMPT = """You are Alexandria the Scribe, a document-focused AI assistant powered by DeepSeek-R1.
You help users find, create, and revise documents.

You have complete autonomy in deciding how to respond to user queries:

1. For greetings or casual conversation, you can respond naturally without using tools.
2. For document-related requests, you should use the appropriate tools to find or manage documents.
3. For complex document tasks, you may need to use multiple tools to provide a comprehensive solution.

WHEN TO CREATE KNOWLEDGE ARTIFACTS:
- When a user asks you to save information as an article, document, or artifact
- When you've revised a document and the user wants to save the changes as a new artifact
- When you want to preserve important information from a conversation in a structured format
- When creating reference materials based on document search results

WORKFLOW EXAMPLE - CREATING A NEW ARTIFACT FROM DOCUMENT SEARCH:
1. User asks: "Can you create an article about our marketing strategy based on our documents?"
2. You search for relevant documents:
   <tool>{"tool_name": "search_documents", "parameters": {"query": "marketing strategy"}}</tool>
3. After analyzing the search results, you organize the information and create a new artifact:
   <tool>{"tool_name": "create_artifact", "parameters": {"title": "Company Marketing Strategy Overview", "author": "Alexandria", "content": "<h1>Company Marketing Strategy Overview</h1>\n<h2>Key Objectives</h2>\n<p>Based on company documents, our marketing strategy focuses on...</p>"}}</tool>

FORMAT YOUR RESPONSES USING HTML:
- Use <h1> for document title (required at beginning of articles)
- Use <h2> for section titles
- Use <h3> for subsection headings
- Use <ul> and <li> for bullet point lists
- Use <ol> and <li> for numbered lists
- Use <b> for emphasis on important points
- Use <i> for definitions or specialized terms
- Use <blockquote> for quoted content
- Use <hr> for separating major sections
- Use <em> to highlight discrepancies or warnings
DO NOT use color styling tags (like style="color:red").

You have access to the following tools:

- search_knowledge_base
- search_documents
- search_artifacts
- list_artifacts
- get_artifact
- create_artifact
- search_web
- calculate
- summarize_text

To use a tool, format your response like this:

<tool>
{
  "tool_name": "search_artifacts",
  "parameters": {
    "query": "example search query"
  }
}
</tool>

Remember that you are an autonomous agent - you decide when to use tools and when to respond directly. Be conversational and helpful, and use your best judgment to determine the appropriate response for each user query.
"""

current_prompt_template = DEFAULT_WISE_PROMPT
current_agent_type = "wise"

# Global LLM instance
_global_llm = None

@lru_cache(maxsize=1)
def get_global_llm():
    """Get or create the global LLM instance."""
    global _global_llm
    if _global_llm is None:
        try:
            # Check if we have an OpenRouter API key
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_api_key:
                _global_llm = OpenRouterWrapper(
                    api_key=openrouter_api_key,
                    model="deepseek/deepseek-r1:free",
                    temperature=0
                )
                logger.info("Using OpenRouter with DeepSeek-R1 model")
            else:
                # Fall back to Azure if OpenRouter key not available
                if not github_token:
                    raise ValueError("No API credentials found (OPENROUTER_API_KEY or GITHUB_TOKEN)")

                # If you have an Azure wrapper, place it here; below is a placeholder:
                _global_llm = AzureLlamaWrapper(
                    endpoint=os.getenv("AZURE_INFERENCE_ENDPOINT", "https://models.inference.ai.azure.com"),
                    model="DeepSeek-R1",
                    credential=AzureKeyCredential(github_token),
                    temperature=0
                )
                logger.info("Using Azure with DeepSeek-R1 model")

        except Exception as e:
            logger.exception("Error creating AI model", error=str(e))

            # Rate limited fallback
            if "429" in str(e).lower() or "rate limit" in str(e).lower():
                class RateLimitedLLM:
                    def invoke(self, *args, **kwargs):
                        return AIMessage(
                            content="I apologize, but the API is currently rate-limited. Please try again later."
                        )
                _global_llm = RateLimitedLLM()
            else:
                class MockLLM:
                    def invoke(self, *args, **kwargs):
                        return AIMessage(
                            content="I'm having trouble connecting to my AI knowledge source right now. "
                                    "Please check your configuration."
                        )
                _global_llm = MockLLM()

    return _global_llm

def create_llm(model=None):
    """Get the global LLM instance."""
    return get_global_llm()

@lru_cache(maxsize=32)
def get_formatted_tools(agent_type):
    """Get formatted tool strings for prompts."""
    tools_list = get_tools(agent_type)
    tool_strings = []
    for tool in tools_list:
        tool_strings.append(
            f"""<tool_description>
name: {tool.name}
description: {tool.description}
parameters: {json.dumps(tool.args, indent=2)}
</tool_description>"""
        )
    return "\n".join(tool_strings)

tool_instructions = """
When using tools:
1. For simple questions, respond directly without tools
2. For complex queries, use appropriate tools
3. Format tool calls properly with <tool> tags
4. Wait for tool results before continuing
5. Provide clear, concise answers based on tool results
"""

@lru_cache(maxsize=32)
def get_system_messages(agent_type):
    """Get cached system messages for the agent type."""
    if agent_type == "scribe":
        system_prompt = DEFAULT_SCRIBE_PROMPT
    else:
        system_prompt = DEFAULT_WISE_PROMPT

    formatted_tools = get_formatted_tools(agent_type)

    return [
        SystemMessage(content=system_prompt),
        SystemMessage(content=f"Available tools:\n\n{formatted_tools}"),
        SystemMessage(content=tool_instructions),
    ]


# -------------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------------
@tool
def search_knowledge_base(query: str, search_type: str = "all") -> str:
    """
    Search the user's personal knowledge base for information related to 'query'.
    """
    try:
        return search_knowledge_base(query, search_type=search_type, config={})
    except TypeError:
        return search_knowledge_base(query, search_type=search_type)

@tool
def search_documents(query: str) -> str:
    """Search uploaded documents for information matching 'query'."""
    try:
        return kb_search_documents(query, config={})
    except TypeError:
        return kb_search_documents(query)

@tool
def search_artifacts(query: str) -> str:
    """Search knowledge artifacts for information matching 'query'."""
    try:
        return kb_search_artifacts(query, config={})
    except TypeError:
        return kb_search_artifacts(query)

@tool
def search_web(query: str, num_results: int = 3) -> str:
    """
    Search the web for information related to 'query'.
    Args:
        query: The search query string
        num_results: Number of results to retrieve
    """
    try:
        return web_search_tool(query, num_results, config={})
    except TypeError:
        return web_search_tool(query, num_results)

@tool
def summarize_text(text: str) -> str:
    """Summarize a long piece of text into a concise form."""
    logger.debug("Summarizing text", text_preview=text[:50])
    try:
        llm = get_global_llm()
        response = llm.invoke(
            f"Please summarize the following text concisely:\n\n{text}"
        )
        logger.debug("Summarize LLM response", response_preview=response.content[:70])
        return response.content
    except Exception as e:
        logger.exception("Error summarizing text", error=str(e))
        return f"Error summarizing text: {str(e)}"

@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression (e.g., "2+2", "sin(30)", "sqrt(16)").
    Returns the result as a string.
    """
    if re.search(r"[^0-9\s\+\-\*\/\(\)\.\,\^\%\=a-zA-Z]", expression):
        return "Error: Invalid characters in expression."

    expression = expression.replace("^", "**")
    safe_dict = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "pi": math.pi, "e": math.e
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except Exception as e:
        return f"Error calculating result: {str(e)}"

@tool
def list_artifacts() -> str:
    """List all currently available knowledge artifacts."""
    from app import artifacts
    if not artifacts:
        return "No artifacts found."
    result = "Available artifacts:\n\n"
    for artifact_id, artifact in artifacts.items():
        result += f"{artifact_id}. {artifact['title']} (by {artifact['author']})\n"
    return result

@tool
def get_artifact(artifact_name: str) -> str:
    """
    Retrieve a specific artifact by name or ID.
    Args:
        artifact_name: The ID or substring of the artifact's title
    """
    from app import artifacts
    try:
        artifact_id = int(artifact_name)
        if artifact_id in artifacts:
            artifact = artifacts[artifact_id]
            return (
                f"Title: {artifact['title']}\nAuthor: {artifact['author']}\n\n"
                f"Content:\n{artifact['content']}"
            )
    except ValueError:
        pass

    for aid, artifact in artifacts.items():
        if artifact_name.lower() in artifact["title"].lower():
            return (
                f"Title: {artifact['title']}\nAuthor: {artifact['author']}\n\n"
                f"Content:\n{artifact['content']}"
            )

    return f"No artifact found with name or ID: {artifact_name}"

@tool
def revise_document(query, document_query):
    """
    Uses the Scribe Agent to revise a document based on the specified query.
    After revision, you can save the document as a new artifact using create_artifact.

    Args:
        query: The specific changes to make to the document
        document_query: A query to identify which document to modify

    Returns:
        The fully revised document with HTML formatting
    """
    logger.info(f"Received document revision request: {query}")
    try:
        current_message = ""
        if "current_message" in globals() and globals()["current_message"]:
            current_message = globals()["current_message"]
        elif "last_message" in globals() and globals()["last_message"]:
            current_message = globals()["last_message"]
        else:
            current_message = query

        # Create a message with the delegation format for the multi-agent architecture
        delegated_message = f"""
I need to revise a document. Here are the details:

<delegate_document_revision>
document_query: "{document_query}"
revision_query: "{query}"
</delegate_document_revision>

{current_message}
        """
        
        # Use the multi-agent graph with the Wise Agent, which will delegate to Scribe
        messages = [HumanMessage(content=delegated_message)]
        initial_state = {
            "messages": messages,
            "current_agent": "wise",
            "document_query": document_query,
            "revision_query": query,
            "revised_document": None
        }
        
        # Invoke the multi-agent graph
        try:
            result = multi_agent_graph.invoke(initial_state)
            
            # Get the revised document from the result
            revised_document = result.get("revised_document", "")
            if revised_document and len(revised_document) > 100:
                if not revised_document.strip().startswith("<h1>"):
                    revised_document = (
                        "<p><em>Note: The revised document should begin with an &lt;h1&gt; "
                        "tag for proper formatting.</em></p>\n\n" + revised_document
                    )
                return revised_document
            
            # If no revised document, check the last AI message
            messages = result.get("messages", [])
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            if ai_messages:
                last_ai_message = ai_messages[-1].content
                # Extract the document from the message if possible
                pattern = r"Here's the updated version:\s*\n\n(.*?)(?:\n\n<think>|\n\n<p><em>Note:|$)"
                match = re.search(pattern, last_ai_message, re.DOTALL)
                if match:
                    return match.group(1)
                return last_ai_message
        except Exception as e:
            logger.exception(f"Error in multi-agent graph: {str(e)}")
            # Fall back to the old implementation
            response = scribe_agent_prompt_chain(
                message=current_message,
                document_query=document_query,
                query=query
            )
            
            if response and isinstance(response, str) and len(response) > 100:
                if not response.strip().startswith("<h1>"):
                    response = (
                        "<p><em>Note: The revised document should begin with an &lt;h1&gt; "
                        "tag for proper formatting.</em></p>\n\n" + response
                    )
                return response

        return (
            "The Scribe Agent was unable to revise the document. Please try again with more specific information "
            "about which document to modify and what changes to make."
        )
    except Exception as e:
        logger.error(f"Error in revise_document: {e}")
        return f"Error revising document: {str(e)}"

@tool
def create_artifact(title: str, author: str, content: str) -> str:
    """
    Create a new knowledge artifact with the specified title, author, and content.
    The content should be in HTML format following the standard article structure.
    This will be added to the knowledge base as a permanent artifact.
    """
    content = clean_html_for_artifact(content, title)
    artifact_data = {"title": title, "author": author, "content": content}
    return f'<create_artifact>{json.dumps(artifact_data)}</create_artifact>'


# -------------------------------------------------------------------------
# Tools selection logic
# -------------------------------------------------------------------------
@lru_cache(maxsize=32)
def get_tools(agent_type):
    """Return the list of tools available for a given agent type."""
    available_tools = [
        search_knowledge_base,
        search_documents,
        search_artifacts,
        search_web,
        calculate,
        summarize_text,
        list_artifacts,
        get_artifact,
        create_artifact,
        revise_document
    ]

    # For the Wise agent, we want all tools except revise_document (since we use direct agent delegation now)
    if agent_type == "wise":
        # For Wise agent, exclude revise_document as we'll use direct delegation instead
        return [t for t in available_tools if t != revise_document]
    elif agent_type == "scribe":
        return available_tools
    elif agent_type == "research":
        # Exclude some specialized tools
        return [t for t in available_tools if t not in [list_artifacts, get_artifact, create_artifact, revise_document]]
    return [
        search_knowledge_base,
        search_documents,
        search_artifacts,
        search_web,
        calculate,
        summarize_text
    ]


# -------------------------------------------------------------------------
# Greeting Detector
# -------------------------------------------------------------------------
def handle_greeting(query: str) -> Optional[str]:
    """
    If 'query' is a basic greeting (hi, hello, etc.), return a greeting response;
    otherwise return None.
    """
    clean_query = query.lower().strip().rstrip("!?.,:;")
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "hola"]

    if clean_query in greetings:
        return "Hello! I'm Alexandria, your knowledge assistant. How can I help you today?"

    for greeting in greetings:
        if clean_query.startswith(greeting):
            remaining = clean_query[len(greeting):].strip()
            if not remaining or remaining in ["there", "alexandria", "everyone", "all", "friend", "friends"]:
                return "Hello! I'm Alexandria, your knowledge assistant. How can I help you today?"

    common_phrases = ["how are you", "how's it going", "what's up", "how do you do", "nice to meet you"]
    for phrase in common_phrases:
        if phrase in clean_query:
            return ("Hello! I'm doing well, thank you for asking. I'm Alexandria, your knowledge assistant. "
                    "How can I help you today?")
    return None


# -------------------------------------------------------------------------
# Building an Agent with a StateGraph Workflow
# -------------------------------------------------------------------------
def agent_executor(messages, agent_type="research", config=None):
    """
    Execute the agent with the given messages.

    Args:
        messages: List of messages to process
        agent_type: Type of agent to use
        config: Additional configuration

    Returns:
        Updated list of messages after agent processing
    """
    if config is None:
        config = {}

    llm = create_llm()
    executor = create_agent_executor(llm, agent_type, config)

    tools = get_tools(agent_type)
    tools_dict = {t.name: t for t in tools}

    state = {
        "messages": messages,
        "tools": tools_dict,
        "tool_calls": [],
        "last_error": None
    }

    logger.info("Executing agent", agent_type=agent_type, num_messages=len(messages))
    max_iterations = config.get("max_iterations", 25)

    try:
        result = executor.invoke(state, {"recursion_limit": max_iterations})
        if result.get("last_error"):
            logger.error("Agent execution error", error=result["last_error"])
            messages = result["messages"]
            error_message = (
                f"I encountered an error: {result['last_error']}. Please try again or rephrase your question."
            )
            error_message = (
                f"{error_message}\n<think>Error encountered during agent execution: "
                f"{result['last_error']}</think>"
            )
            messages.append(AIMessage(content=error_message))
        else:
            messages = result["messages"]
            ai_messages = [i for i, m in enumerate(messages) if isinstance(m, AIMessage)]
            if ai_messages:
                last_ai_index = ai_messages[-1]
                last_content = messages[last_ai_index].content
                if "<think>" not in last_content:
                    logger.info("Adding thinking section to agent executor response")
                    detailed_thinking = "Agent Execution Process:\n\n"
                    detailed_thinking += "1. REQUEST ANALYSIS\n"
                    detailed_thinking += "   • Processed user request using agent framework\n"
                    detailed_thinking += f"   • Used agent type: {agent_type}\n\n"
                    detailed_thinking += "2. RESPONSE FORMULATION\n"
                    detailed_thinking += "   • Analyzed available information\n"
                    detailed_thinking += "   • Generated appropriate response based on context\n"

                    updated_content = f"{last_content}\n<think>{detailed_thinking}</think>"
                    messages[last_ai_index] = AIMessage(content=updated_content)
        return messages
    except Exception as e:
        logger.exception("Agent execution failed", error=str(e))
        error_message = (
            f"I encountered an unexpected error: {str(e)}. "
            f"Please try again or contact support if the issue persists."
        )
        error_message = (
            f"{error_message}\n<think>Exception in agent_executor: {str(e)}</think>"
        )
        messages.append(AIMessage(content=error_message))
        return messages


@lru_cache(maxsize=1000)
def _cached_tool_execution(tool_name: str, tool_args_str: str):
    """Cache tool execution results to avoid redundant calls."""
    try:
        tools_list = get_tools(current_agent_type)
        tools_dict = {t.name: t for t in tools_list}

        if tool_name not in tools_dict:
            return f"<tool_result>Error: Tool '{tool_name}' not found</tool_result>"

        tool = tools_dict[tool_name]
        tool_args = json.loads(tool_args_str)

        if hasattr(tool, "invoke"):
            if "query" in tool_args and len(tool_args) == 1:
                result = tool.invoke(tool_args["query"])
            else:
                result = tool.invoke(tool_args)
        else:
            if hasattr(tool, "run"):
                if "query" in tool_args and len(tool_args) == 1:
                    result = tool.run(tool_args["query"])
                else:
                    result = tool.run(**tool_args)
            else:
                result = tool(**tool_args)

        return f"<tool_result>{result}</tool_result>"
    except Exception as e:
        logger.exception("Error in cached tool execution", error=str(e))
        return f"<tool_result>Error: {str(e)}</tool_result>"


def execute_tool(tool_name, tool_input, available_tools):
    """Execute a tool with caching for better performance."""
    try:
        if isinstance(tool_input, dict):
            tool_input_str = json.dumps(tool_input, sort_keys=True)
        else:
            tool_input_str = str(tool_input)
        return _cached_tool_execution(tool_name, tool_input_str)
    except Exception as e:
        logger.exception("Error executing tool", tool_name=tool_name, error=str(e))
        return f"<tool_result>Error executing tool '{tool_name}': {str(e)}</tool_result>"


def parse_ai_message_for_tools(state):
    """Parse the last AI message for tool calls and extract them."""
    messages = state["messages"]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]

    if not ai_messages:
        logger.warning("No AI messages found in state")
        return {"tool_calls": []}

    last_ai_message = ai_messages[-1]
    tool_calls = []

    tool_blocks = re.findall(r"<tool>(.*?)</tool>", last_ai_message.content, re.DOTALL)
    for tool_block in tool_blocks:
        try:
            tool_data = json.loads(tool_block)
            tool_calls.append({
                "name": tool_data.get("tool_name"),
                "arguments": tool_data.get("parameters", {})
            })
        except json.JSONDecodeError:
            logger.error("Failed to parse tool block", tool_block=tool_block)
            continue

    logger.info("Parsed tool calls", num_calls=len(tool_calls))
    return {"tool_calls": tool_calls}


def tool_execution_node(state):
    """Execute tools and update state with results."""
    tools = state["tools"]
    tool_calls = state["tool_calls"]
    messages = state["messages"]
    tool_results = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        result = execute_tool(tool_name, tool_args, tools)
        tool_results.append(result)

    # Update the last AI message content with tool results
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            content = messages[i].content
            for idx, tool_block in enumerate(
                re.findall(r"<tool>(.*?)</tool>", content, re.DOTALL)
            ):
                if idx < len(tool_results):
                    pattern = f"<tool>{re.escape(tool_block)}</tool>"
                    content = re.sub(pattern, tool_results[idx], content, count=1)
            messages[i] = AIMessage(content=content)
            break

    return {"messages": messages}


def build_graph(agent_type="wise"):
    """Build the agent graph using LangGraph."""
    logger.info("Building graph", agent_type=agent_type)
    global current_agent_type
    current_agent_type = agent_type

    llm = get_global_llm()
    tools_list = get_tools(agent_type)
    tools_dict = {t.name: t for t in tools_list}
    system_messages = get_system_messages(agent_type)

    workflow = StateGraph(AgentState)

    def agent_executor_node(state):
        """Execute the agent with the current state."""
        state_messages = state["messages"]
        logger.info("Processing messages", num_messages=len(state_messages))

        last_user_msg = None
        iteration_count = 0
        filtered_messages = []

        for msg in state_messages:
            if isinstance(msg, HumanMessage) and last_user_msg is None:
                last_user_msg = msg.content
            if isinstance(msg, SystemMessage) and msg.content.startswith("Iteration count:"):
                try:
                    iteration_count = int(msg.content.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif not (
                isinstance(msg, SystemMessage)
                and (
                    msg.content.startswith("Iteration count:")
                    or msg.content.startswith("IMPORTANT: You've used multiple tools already")
                    or msg.content.startswith("You have made several tool calls")
                )
            ):
                filtered_messages.append(msg)

        iteration_count += 1

        if iteration_count >= 8:
            logger.info("Reached iteration threshold, encouraging direct answer", iteration_count=iteration_count)
            filtered_messages.append(
                SystemMessage(content=(
                    "IMPORTANT: You've used multiple tools already. For efficiency, try to provide a direct "
                    "answer now based on what you already know. Only use additional tools if absolutely necessary."
                ))
            )

        if iteration_count >= 15:
            logger.info("Reached high iteration count, forcing direct answer", iteration_count=iteration_count)
            filtered_messages.append(
                SystemMessage(content=(
                    "You have made several tool calls without reaching a conclusion. "
                    "Please answer the user's question directly now without using any more tools. "
                    "Use your existing knowledge to provide the best possible answer."
                ))
            )

        filtered_messages.append(SystemMessage(content=f"Iteration count: {iteration_count}"))

        all_messages = system_messages + filtered_messages

        if iteration_count >= 30:
            logger.warning("Maximum iterations reached, ending conversation", iteration_count=iteration_count)
            error_message = AIMessage(
                content=(
                    "I've done extensive research on your question, but need to provide an answer now. "
                    "Based on the information I've gathered, here's what I can tell you."
                )
            )
            filtered_messages.append(error_message)
            return {"messages": filtered_messages, "next": END}

        ai_message = llm.invoke(all_messages)
        filtered_messages.append(ai_message)

        if "<tool>" in ai_message.content:
            logger.info("Tool call detected, routing to decision node")
            return {"messages": filtered_messages, "next": "decide"}
        else:
            logger.info("No tool call detected, ending workflow")
            return {"messages": filtered_messages, "next": END}

    def entry_point_fn(state):
        """Entry point for the workflow that processes messages."""
        messages = state["messages"]
        if not messages:
            return {"messages": messages, "next": END}
        logger.info("Starting workflow", num_messages=len(messages))
        return {"messages": messages, "next": "agent_executor"}

    def decide_node(state):
        """Decide whether to execute tools or return the response."""
        messages = state["messages"]
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        if not ai_messages:
            logger.warning("No AI messages in state, ending workflow")
            return {"next": "end"}

        last_ai_message = ai_messages[-1]
        if "<tool>" in last_ai_message.content:
            logger.info("Found tool call in AI message, routing to parse_tools")
            return {"next": "parse_tools"}
        logger.info("No tool calls found, ending conversation")
        return {"next": "end"}

    workflow.add_node("entry_point", entry_point_fn)
    workflow.add_node("agent_executor", agent_executor_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("parse_tools", parse_ai_message_for_tools)
    workflow.add_node("execute_tools", tool_execution_node)

    workflow.add_edge("entry_point", "agent_executor")
    workflow.add_edge("agent_executor", "decide")

    workflow.add_conditional_edges(
        "decide",
        lambda x: x["next"],
        {
            "parse_tools": "parse_tools",
            "end": END
        }
    )

    workflow.add_edge("parse_tools", "execute_tools")
    workflow.add_edge("execute_tools", "agent_executor")
    workflow.set_entry_point("entry_point")

    return workflow.compile()

compiled_workflow = build_graph("wise")

def update_agent(new_agent_type, new_prompt_template=None):
    """
    Update the agent type and prompt template, then rebuild the workflow.
    """
    global current_agent_type, current_prompt_template, compiled_workflow
    print(f"[DEBUG update_agent] Called with type={new_agent_type}, custom_prompt={bool(new_prompt_template)}")

    if new_agent_type not in ["wise", "scribe"]:
        new_agent_type = "wise"

    current_agent_type = new_agent_type
    if new_prompt_template:
        current_prompt_template = new_prompt_template
    else:
        if new_agent_type == "scribe":
            current_prompt_template = DEFAULT_SCRIBE_PROMPT
        else:
            current_prompt_template = DEFAULT_WISE_PROMPT

    compiled_workflow = build_graph(current_agent_type)
    return {"status": "success", "message": f"Agent updated to {new_agent_type}"}


def get_agent_response(messages, config=None):
    """
    Get a response from the agent using the provided messages.

    Args:
        messages: List of messages to process
        config: Configuration options

    Returns:
        The agent's response
    """
    if config is None:
        config = {}

    logger.info("Getting agent response", num_messages=len(messages))

    latest_message = None
    chat_history = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            latest_message = msg.content
        chat_history.append(msg)

    # Additional logic for artifact saving or revision could go here...

    agent_type = config.get("agent_type", "research")
    max_iterations = config.get("max_iterations", 10)
    max_tool_uses = config.get("max_tool_uses", 5)

    try:
        updated_messages = agent_executor(
            messages,
            agent_type=agent_type,
            config={
                "max_iterations": max_iterations,
                "max_tool_uses": max_tool_uses
            }
        )
        # Return the last AI message to the user
        ai_messages = [m for m in updated_messages if isinstance(m, AIMessage)]
        if not ai_messages:
            return AIMessage(content="I'm sorry, I wasn't able to generate a response.")
        return ai_messages[-1]
    except Exception as e:
        logger.exception(f"Error in get_agent_response: {str(e)}")
        return AIMessage(
            content=(
                f"I'm having trouble generating a response right now. Error: {str(e)}\n"
                f"<think>Encountered an error during processing: {str(e)}</think>"
            )
        )


# -------------------------------------------------------------------------
# Scribe Agent prompt chaining
# -------------------------------------------------------------------------
def scribe_agent_prompt_chain(message, document_query, query):
    """
    A specialized prompt chaining function for document revision.
    """
    logger.info(f"Starting scribe_agent_prompt_chain with query: {query}")
    try:
        # Stage 1: Identify the document
        stage1_prompt = (
            f"""As the Scribe Agent specializing in document management, you need to identify the correct document based on the user's query.

USER REQUEST: {message}

DOCUMENT QUERY: {document_query}

First, analyze which document is being referenced. Search your knowledge to find the most relevant document that matches this query.

Return your response in the following JSON format:
{{
    "document_title": "Exact title",
    "document_description": "Brief description",
    "confidence": "High/Medium/Low"
}}

If you cannot identify a specific document with at least medium confidence, respond with:
{{
    "document_title": "Unknown",
    "document_description": "Could not identify a specific document",
    "confidence": "Low"
}}"""
        )
        llm = get_global_llm()
        stage1_response = llm.invoke(stage1_prompt, max_tokens=1024)
        stage1_response = stage1_response.content if hasattr(stage1_response, "content") else str(stage1_response)
        logger.info(f"Stage 1 Response: {stage1_response}")

        document_info = {}
        match = re.search(r"({.*})", stage1_response, re.DOTALL)
        if match:
            try:
                document_info = json.loads(match.group(1))
            except Exception as e:
                logger.error(f"Error parsing document identification JSON: {e}")
                document_info = {"document_title": "Unknown", "confidence": "Low"}
        else:
            document_info = {"document_title": "Unknown", "confidence": "Low"}

        document_title = document_info.get("document_title", "Unknown")
        confidence = document_info.get("confidence", "Low")

        if document_title == "Unknown" or confidence == "Low":
            return (
                "I couldn't identify a specific document to revise. "
                "Could you provide more details about the document?"
            )

        # Stage 2: Identify changes
        stage2_prompt = (
            f"""As the Scribe Agent specializing in document management, identify the specific modifications requested.

DOCUMENT TITLE: {document_title}
USER REQUEST: {message}
MODIFICATION QUERY: {query}

Return your response in the following JSON:
{{
    "changes_needed": [
        {{
            "type": "add/remove/modify/format",
            "location": "Where in the document",
            "description": "What changes to make"
        }}
    ],
    "additional_context": "Any additional context"
}}"""
        )
        stage2_response = llm.invoke(stage2_prompt, max_tokens=1024)
        stage2_response = stage2_response.content if hasattr(stage2_response, "content") else str(stage2_response)
        logger.info(f"Stage 2 Response: {stage2_response}")

        changes_info = {}
        match = re.search(r"({.*})", stage2_response, re.DOTALL)
        if match:
            try:
                changes_info = json.loads(match.group(1))
            except Exception as e:
                logger.error(f"Error parsing changes JSON: {e}")

        # Stage 3: Apply changes
        stage3_prompt = (
            f"""As the Scribe Agent specializing in document management, now create the FULL revised version of the document.

DOCUMENT TITLE: {document_title}
USER REQUEST: {message}
MODIFICATION QUERY: {query}

IDENTIFIED CHANGES:
{json.dumps(changes_info, indent=2)}

Return ONLY the complete HTML document with all changes applied:
1. Start with <h1> containing the document title
2. Use <h2>, <h3>, <p>, <ul>, <li>, <blockquote>, <b>, <i> as needed
3. No additional commentary or JSON. Only the final revised document in HTML.
"""
        )
        final_response = llm.invoke(stage3_prompt, max_tokens=16384)
        final_response = final_response.content if hasattr(final_response, "content") else str(final_response)
        logger.info(f"Final response length: {len(final_response)}")

        cleaned_response = re.sub(r"<tool>.*?</tool>", "", final_response, flags=re.DOTALL)
        cleaned_response = re.sub(r"```html|```", "", cleaned_response, flags=re.DOTALL)

        # Add metadata for revision
        thinking_details = {
            "document_identified": document_title,
            "confidence": confidence,
            "changes_required": changes_info
        }
        return f"{cleaned_response}\n\n<revision_metadata>{json.dumps(thinking_details)}</revision_metadata>"

    except Exception as e:
        logger.error(f"Error in scribe_agent_prompt_chain: {e}")
        return f"I encountered an error while trying to revise the document: {str(e)}"


def chat_with_agent(message, chat_history=None, agent_type="research"):
    """
    Chat with the agent - using the new multi-agent architecture
    """
    messages = []
    
    # Process chat history
    if chat_history:
        if isinstance(chat_history, str):
            lines = chat_history.split("\n")
            for line in lines:
                if line.startswith("Human: "):
                    content = line[7:]
                    if content.strip():
                        messages.append(HumanMessage(content=content))
                elif line.startswith("AI: "):
                    content = line[4:]
                    if content.strip():
                        messages.append(AIMessage(content=content))
        elif isinstance(chat_history, list):
            for msg in chat_history:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user" and content:
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant" and content:
                        messages.append(AIMessage(content=content))
    
    # Add the current message
    if message and message.strip():
        messages.append(HumanMessage(content=message))
    
    # Initialize state
    initial_state = {
        "messages": messages,
        "current_agent": "wise",
        "document_query": None,
        "revision_query": None,
        "revised_document": None
    }
    
    try:
        # Use the multi-agent graph
        if agent_type.lower() == "scribe":
            # If specifically requesting Scribe agent, start with that
            initial_state["current_agent"] = "scribe"
            result = multi_agent_graph.invoke(initial_state, {"agent": "scribe_agent"})
        else:
            # Otherwise start with Wise agent
            result = multi_agent_graph.invoke(initial_state)
        
        # Extract the final messages
        final_messages = result.get("messages", [])
        
        # Get the last AI message
        ai_messages = [msg for msg in final_messages if isinstance(msg, AIMessage)]
        if ai_messages:
            last_ai_message = ai_messages[-1]
            return {"response": last_ai_message.content}
        else:
            return {"response": "I'm sorry, I wasn't able to generate a response."}
            
    except Exception as e:
        logger.exception(f"Error in chat_with_agent: {str(e)}")
        return {"response": f"I encountered an error: {str(e)}"}


def create_agent_executor(llm, agent_type="research", config=None):
    """Create an agent executor using the LangGraph standard pattern."""
    tools = get_tools(agent_type)
    tools_dict = {t.name: t for t in tools}

    tool_strings = []
    for tool in tools:
        tool_strings.append(
            f"""<tool_description>
name: {tool.name}
description: {tool.description}
parameters: {json.dumps(tool.args, indent=2)}
</tool_description>"""
        )
    tools_string = "\n".join(tool_strings)

    if config is None:
        config = {}

    system_prompt = f"""You are an AI assistant that can answer questions about the Alexandria documentation.
You have access to the following tools:

{tools_string}

FORMAT YOUR RESPONSES USING HTML:
- Use <h2> for section titles
- Use <h3> for subsection headings
- Use <ul> and <li> for bullet point lists
- Use <ol> and <li> for numbered lists
- Use <b> for emphasis on important points
- Use <i> for definitions or specialized terms
- Use <blockquote> for quoted content
- Use <hr> for separating major sections
- Use <em> to highlight discrepancies or warnings
DO NOT use color styling tags (like style="color:red").

To use a tool, use the following format:
<tool>
{{
  "tool_name": "<name of the tool>",
  "parameters": {{
    "<parameter name>": "<parameter value>"
  }}
}}
</tool>

The tool result will be returned as:
<tool_result>
result
</tool_result>

Answer questions as helpfully as possible. For simple questions, answer directly. For complex queries, use tools.
If you don't know the answer, say so.
"""

    def llm_node(state):
        """Generate a response using the LLM."""
        try:
            messages = state["messages"]
            logger.info("Calling LLM", num_messages=len(messages))

            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=system_prompt)] + messages

            last_error = state.get("last_error")
            rate_limited = False
            if last_error and isinstance(last_error, str):
                rate_limited = "rate limit" in last_error or "429" in last_error

            if rate_limited:
                logger.warning("Using fallback response due to rate limiting")
                fallback_content = (
                    "I apologize, but the API is currently rate-limited. Please try again later."
                )
                response = AIMessage(content=fallback_content)
            else:
                response = llm(messages)

            updated_messages = messages + [AIMessage(content=response.content)]
            return {"messages": updated_messages}
        except Exception as e:
            logger.exception("Error in LLM node", error=str(e))
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                return {"messages": state["messages"], "last_error": "Rate limiting detected"}
            return {"messages": state["messages"], "last_error": str(e)}

    def decide_node(state):
        """Decide whether to execute tools or return the response."""
        messages = state["messages"]
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        if not ai_messages:
            logger.warning("No AI messages in state, ending workflow")
            return {"next": "end"}

        last_ai_message = ai_messages[-1]
        if "<tool>" in last_ai_message.content:
            logger.info("Found tool call in AI message, routing to tools")
            return {"next": "call_tools"}
        return {"next": "end"}

    workflow = StateGraph(AgentState)
    workflow.add_node("llm", llm_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("call_tools", parse_ai_message_for_tools)
    workflow.add_node("execute_tools", tool_execution_node)

    workflow.add_edge("llm", "decide")

    workflow.add_conditional_edges(
        "decide",
        lambda x: x["next"],
        {
            "call_tools": "call_tools",
            "end": END
        }
    )

    workflow.add_edge("call_tools", "execute_tools")
    workflow.add_edge("execute_tools", "llm")
    workflow.set_entry_point("llm")

    return workflow.compile()


def clean_html_for_artifact(content, title=None):
    """
    Clean and format HTML content to ensure it's suitable for saving as an artifact.
    """
    content = content.strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"<tool_result>.*?</tool_result>", "", content, flags=re.DOTALL)
    content = re.sub(
        r"<p><em>You can save this revised document as a new artifact.*?</em></p>", "",
        content, flags=re.DOTALL
    )
    content = re.sub(
        r"<p><em>Note: The revised document should begin with.*?</em></p>", "", content,
        flags=re.DOTALL
    )

    if not re.match(r"^\s*<h1>", content):
        if title:
            content = f"<h1>{title}</h1>\n\n{content}"
        else:
            first_line = re.search(r"^(.*?)[\n\r<]", content)
            if first_line:
                extracted_title = first_line.group(1).strip()
                if extracted_title and len(extracted_title) < 100 and not extracted_title.startswith("<"):
                    content = f"<h1>{extracted_title}</h1>\n\n{content}"
                else:
                    content = f"<h1>Untitled Document</h1>\n\n{content}"
            else:
                content = f"<h1>Untitled Document</h1>\n\n{content}"

    if not re.search(r"<p>", content):
        content = re.sub(r"(\n\n+)(?!<)", r"\1<p>", content)
        content = re.sub(r"(?<!>)\n\n+", "</p>\n\n", content)
    return content


def extract_artifact_details_from_message(message, default_title="Untitled Document"):
    """
    Extract artifact details (title, author, etc.) from a user message.
    """
    details = {
        "title": default_title,
        "author": "Alexandria",
        "date": None,
        "category": None,
        "tags": []
    }
    if not message or not isinstance(message, str):
        return details

    msg_lower = message.lower()

    title_patterns = [
        r'title:?\s*["\'"]?([^"\'\.,;]+)["\'"]?',
        r'name(?:d|)\s+(?:it|the document|the artifact):?\s*["\'"]?([^"\'\.,;]+)["\'"]?',
        r'call(?:ed|)\s+(?:it|the document|the artifact):?\s*["\'"]?([^"\'\.,;]+)["\'"]?',
        r'save\s+(?:it|this|the document)\s+as\s*["\'"]?([^"\'\.,;]+)["\'"]?'
    ]
    for pattern in title_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            details["title"] = match.group(1).strip().title()
            break

    author_patterns = [
        r'author:?\s*["\'"]?([^"\'\.,;]+)["\'"]?',
        r'by:?\s*["\'"]?([^"\'\.,;]+)["\'"]?',
        r'credit(?:ed)? to:?\s*["\'"]?([^"\'\.,;]+)["\'"]?'
    ]
    for pattern in author_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            details["author"] = match.group(1).strip()
            break

    date_match = re.search(r'date:?\s*["\'"]?([^"\'\.,;]+)["\'"]?', msg_lower)
    if date_match:
        details["date"] = date_match.group(1).strip()

    category_match = re.search(r'category:?\s*["\'"]?([^"\'\.,;]+)["\'"]?', msg_lower)
    if category_match:
        details["category"] = category_match.group(1).strip()

    tags_match = re.search(r'tags?:?\s*["\'"]?([^"\']+)["\'"]?', msg_lower)
    if tags_match:
        tags_str = tags_match.group(1).strip()
        tags = [tag.strip() for tag in re.split(r"[,\s]+", tags_str) if tag.strip()]
        details["tags"] = tags

    return details


def simple_chat_with_agent(message, chat_history=None):
    """
    A simplified approach to chat with the agent.
    """
    logger.info("Starting simple chat interaction")

    if message and isinstance(message, str) and chat_history:
        msg_lower = message.lower()
        if any(
            phrase in msg_lower
            for phrase in ["save as artifact", "make artifact", "create artifact",
                           "save revision", "save document", "make your revision"]
        ):
            logger.info("Detected request to save previous revision as artifact")
            previous_content = None
            previous_title = "Untitled Document"

            if isinstance(chat_history, str):
                lines = chat_history.split("\n")
                for i in range(len(lines) - 1, -1, -1):
                    line = lines[i]
                    if line.startswith("AI: "):
                        content = line[4:]
                        if "<h1>" in content and "</h1>" in content:
                            previous_content = content
                            title_match = re.search(r"<h1>(.*?)</h1>", content)
                            if title_match:
                                previous_title = title_match.group(1)
                            break
            elif isinstance(chat_history, list):
                for msg in chat_history:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if "<h1>" in content and "</h1>" in content:
                            previous_content = content
                            title_match = re.search(r"<h1>(.*?)</h1>", content)
                            if title_match:
                                previous_title = title_match.group(1)
                            break

            if previous_content:
                logger.info(f"Found previous revision: '{previous_title}'")
                details = extract_artifact_details_from_message(message, previous_title)
                try:
                    cleaned_content = clean_html_for_artifact(previous_content, details["title"])
                    artifact_data = {
                        "title": details["title"],
                        "author": details["author"],
                        "content": cleaned_content
                    }
                    special_response = f'<create_artifact>{json.dumps(artifact_data)}</create_artifact>\n'
                    special_response += "<h2>Document Saved as Artifact</h2>\n"
                    special_response += "<p>I've saved the revised document as a new artifact:</p>\n"
                    special_response += (
                        f'<ul>\n<li><b>Title:</b> {details["title"]}</li>\n'
                        f'<li><b>Author:</b> {details["author"]}</li>\n'
                    )
                    if details["category"]:
                        special_response += f'<li><b>Category:</b> {details["category"]}</li>\n'
                    if details["tags"]:
                        special_response += (
                            f'<li><b>Tags:</b> {", ".join(details["tags"])}</li>\n'
                        )
                    special_response += "</ul>\n"
                    special_response += "<p>You can now view and access this document in your artifacts library.</p>"
                    special_response += (
                        f'\n<think>Detected request to save previous revision as artifact. '
                        f'Extracted details: Title="{details["title"]}", Author="{details["author"]}"'
                    )
                    if details["category"]:
                        special_response += f', Category="{details["category"]}"'
                    if details["tags"]:
                        special_response += f', Tags=[{", ".join(details["tags"])}]'
                    special_response += "</think>"
                    return {"response": special_response}
                except Exception as e:
                    logger.error(f"Error creating artifact from previous revision: {e}")

    messages = []
    messages.append(SystemMessage(content=current_prompt_template))

    if chat_history:
        if isinstance(chat_history, str):
            lines = chat_history.split("\n")
            for line in lines:
                if line.startswith("Human: "):
                    content = line[7:]
                    if content.strip():
                        messages.append(HumanMessage(content=content))
                elif line.startswith("AI: "):
                    content = line[4:]
                    if content.strip():
                        messages.append(AIMessage(content=content))
        elif isinstance(chat_history, list):
            for msg in chat_history:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user" and content:
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant" and content:
                        messages.append(AIMessage(content=content))

    if message and message.strip():
        messages.append(HumanMessage(content=message))

    response = get_agent_response(messages, {"agent_type": current_agent_type})
    return {"response": response.content}


# -------------------------------------------------------------------------
# Multi-Agent Architecture Implementation
# -------------------------------------------------------------------------
def wise_agent(state: AgentState) -> Command[Literal["wise_agent", "scribe_agent", END]]:
    """
    The Wise Agent node that can process messages and delegate to Scribe Agent when needed.
    """
    messages = state.get("messages", [])
    llm = get_global_llm()
    
    # Add system message if not already present
    has_system_message = any(isinstance(msg, SystemMessage) for msg in messages)
    if not has_system_message:
        messages.insert(0, SystemMessage(content=DEFAULT_WISE_PROMPT))
    
    # Process with LLM
    logger.info("Processing with Wise Agent", num_messages=len(messages))
    response = llm.invoke(messages)
    
    # Check if document revision is needed
    if "<delegate_document_revision>" in response.content:
        # Extract document revision details using regex
        import re
        doc_match = re.search(r'document_query:\s*"([^"]+)"', response.content)
        revision_match = re.search(r'revision_query:\s*"([^"]+)"', response.content)
        
        document_query = doc_match.group(1) if doc_match else ""
        revision_query = revision_match.group(1) if revision_match else ""
        
        # If we have the necessary information, delegate to Scribe Agent
        if document_query and revision_query:
            # Clean the response to remove the delegation instruction
            cleaned_response = re.sub(r'<delegate_document_revision>.*?</delegate_document_revision>', 
                                    "I'll help with revising this document. Let me work on that for you.", 
                                    response.content, flags=re.DOTALL)
            
            # Create a user-visible response
            messages.append(AIMessage(content=cleaned_response))
            
            logger.info(f"Delegating to Scribe Agent: document='{document_query}', revision='{revision_query}'")
            
            # Return command to hand off to Scribe Agent with document details
            return Command(
                goto="scribe_agent",
                update={
                    "messages": messages,
                    "current_agent": "scribe",
                    "document_query": document_query,
                    "revision_query": revision_query
                }
            )
    
    # If no delegation is needed, update messages and continue or end
    messages.append(AIMessage(content=response.content))
    return Command(
        goto=END,
        update={"messages": messages}
    )

def scribe_agent(state: AgentState) -> Command[Literal["wise_agent", END]]:
    """
    The Scribe Agent node that specializes in document revision.
    """
    document_query = state.get("document_query", "")
    revision_query = state.get("revision_query", "")
    messages = state.get("messages", [])
    
    logger.info(f"Scribe Agent processing: document='{document_query}', revision='{revision_query}'")
    
    # Use the three-stage document revision process
    try:
        # Extract the most recent human message if available
        latest_human_msg = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                latest_human_msg = msg.content
                break
        
        # Implement the three-stage document revision process
        result = scribe_agent_prompt_chain(
            message=latest_human_msg,
            document_query=document_query,
            query=revision_query
        )
        
        # Prepare the Scribe Agent's response with the revised document
        scribe_response = (
            f"I've revised the document as requested. Here's the updated version:\n\n{result}"
        )
        
        # Add thinking metadata if present
        metadata_match = re.search(r"<revision_metadata>(.*?)</revision_metadata>", result, re.DOTALL)
        thinking_content = ""
        if metadata_match:
            try:
                metadata_json = json.loads(metadata_match.group(1))
                result = result.replace(metadata_match.group(0), "")
                thinking_content = "Document Revision Process:\n\n"
                thinking_content += f"1. DOCUMENT IDENTIFICATION\n"
                thinking_content += f"   • Identified document: \"{metadata_json.get('document_identified', 'Unknown')}\"\n"
                thinking_content += f"   • Confidence level: {metadata_json.get('confidence', 'Low')}\n\n"
                changes = metadata_json.get("changes_required", {}).get("changes_needed", [])
                thinking_content += "2. CHANGE ANALYSIS\n"
                if changes:
                    for i, change in enumerate(changes):
                        thinking_content += f"   • Change {i+1}: {change.get('type', 'Unknown')} "
                        thinking_content += f"at {change.get('location', 'Unknown')}\n"
                        thinking_content += f"     Description: {change.get('description', 'N/A')}\n"
                else:
                    thinking_content += "   • No specific changes were identified in JSON format\n"
                additional_context = metadata_json.get("changes_required", {}).get("additional_context")
                if additional_context:
                    thinking_content += f"\n   • Additional context: {additional_context}\n"
                thinking_content += "\n3. DOCUMENT REVISION\n"
                thinking_content += "   • Applied all changes to the document\n"
                
                scribe_response += f"\n\n<think>{thinking_content}</think>"
            except Exception as e:
                logger.error(f"Error processing revision metadata: {e}")
        
        # Add a note about saving as artifact
        scribe_response += (
            "\n\n<p><em>Note: If you'd like to save this revised document as a new artifact, "
            "please let me know, and I'll help you create one.</em></p>"
        )
        
        # Add the response to messages
        messages.append(AIMessage(content=scribe_response))
        
        logger.info("Scribe Agent completed revision, returning to Wise Agent")
        
        # Hand back control to the Wise Agent with the revised document
        return Command(
            goto="wise_agent",
            update={
                "messages": messages,
                "current_agent": "wise",
                "revised_document": result
            }
        )
        
    except Exception as e:
        logger.exception(f"Error in Scribe Agent: {str(e)}")
        error_msg = f"I encountered an error while revising the document: {str(e)}"
        messages.append(AIMessage(content=error_msg))
        
        return Command(
            goto="wise_agent",
            update={
                "messages": messages,
                "current_agent": "wise"
            }
        )

def build_agent_graph():
    """Build a graph with multiple agents that can call each other directly."""
    # Initialize the state graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("wise_agent", wise_agent)
    graph.add_node("scribe_agent", scribe_agent)
    
    # Add edges
    graph.add_edge(START, "wise_agent")
    
    # Compile the graph
    return graph.compile()

# Create and compile the multi-agent graph
multi_agent_graph = build_agent_graph()