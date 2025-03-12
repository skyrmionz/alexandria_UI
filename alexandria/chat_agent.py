import os
import json
import re
from typing import List, Optional, TypedDict, Any

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langchain_community.chat_models import ChatOpenAI

# If these imports reference your local modules, ensure they're correct:
from tools.knowledge_base_tool import knowledge_base_search_tool, search_artifacts as kb_search_artifacts, search_documents as kb_search_documents
from tools.web_search_tool import web_search_tool

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# -------------------------------------------------------------------------
# DEBUG: Print environment info to confirm environment variables are loaded.
# -------------------------------------------------------------------------
print("[DEBUG] Loading environment variables...")
load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")
azure_endpoint = os.getenv("AZURE_ENDPOINT", "https://models.inference.ai.azure.com")
# Set AZURE_OPENAI_ENDPOINT to the same value as AZURE_ENDPOINT for compatibility
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
print(f"[DEBUG] GITHUB_TOKEN present?: {'Yes' if github_token else 'No'}")
print(f"[DEBUG] azure_endpoint = {azure_endpoint}")
print(f"[DEBUG] Set AZURE_OPENAI_ENDPOINT = {azure_endpoint}")

# -------------------------------------------------------------------------
# AgentState typed dict for the LangGraph state.
# -------------------------------------------------------------------------
class AgentState(TypedDict):
    """State for the agent graph."""
    messages: List[BaseMessage]
    next: Optional[str]


# -------------------------------------------------------------------------
# Default prompts. Feel free to adjust as needed.
# -------------------------------------------------------------------------
DEFAULT_WISE_PROMPT = """You are Alexandria, a highly capable AI assistant powered by Meta-Llama-3.1-405B-Instruct.
Your goal is to provide helpful, accurate, and contextually appropriate responses to users.

You have complete autonomy in deciding how to respond to user queries:

1. For greetings or casual conversation, you can respond naturally without using tools.
2. For factual questions or information requests, you should use the appropriate tools to gather information.
3. For complex questions, you may need to use multiple tools to provide a comprehensive answer.

You have access to the following tools:

- search_knowledge_base: Search the user's personal knowledge base
- search_documents: Search through uploaded documents
- search_artifacts: Search through knowledge artifacts
- search_web: Search the internet for information
- calculate: Perform mathematical calculations
- summarize_text: Summarize long pieces of text

To use a tool, format your response like this:

<tool>
{
  "tool_name": "search_knowledge_base",
  "parameters": {
    "query": "example search query"
  }
}
</tool>

Remember that you are an autonomous agent - you decide when to use tools and when to respond directly. Be conversational and helpful, and use your best judgment to determine the appropriate response for each user query.
"""

DEFAULT_SCRIBE_PROMPT = """You are Alexandria the Scribe, a document-focused AI assistant powered by Meta-Llama-3.1-405B-Instruct.
You help users find, create, and revise documents.

You have complete autonomy in deciding how to respond to user queries:

1. For greetings or casual conversation, you can respond naturally without using tools.
2. For document-related requests, you should use the appropriate tools to find or manage documents.
3. For complex document tasks, you may need to use multiple tools to provide a comprehensive solution.

You have access to the following tools:

- search_knowledge_base: Search the user's personal knowledge base
- search_documents: Search through uploaded documents
- search_artifacts: Search through knowledge artifacts
- list_artifacts: List all available knowledge artifacts
- get_artifact: Retrieve a specific artifact by name
- search_web: Search the internet for information
- calculate: Perform mathematical calculations
- summarize_text: Summarize long pieces of text

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


# -------------------------------------------------------------------------
# Global state & tracking variables
# -------------------------------------------------------------------------
current_prompt_template = DEFAULT_WISE_PROMPT
current_agent_type = "wise"


# -------------------------------------------------------------------------
# LLM Initialization
# -------------------------------------------------------------------------
def get_llm():
    """
    Get the LLM instance using Azure AI Inference SDK's ChatCompletionsClient.
    Simple, direct implementation for Meta-Llama models.
    """
    try:
        print("[DEBUG get_llm] Initializing LLM...")
        
        # Get environment variables
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_ENDPOINT")
        github_token = os.environ.get("GITHUB_TOKEN")
        
        if not azure_endpoint or not github_token:
            print("[DEBUG get_llm] Missing required environment variables")
            raise ValueError("Azure endpoint and GitHub token are required")
            
        print(f"[DEBUG get_llm] Using Azure endpoint: {azure_endpoint}")
        print(f"[DEBUG get_llm] GitHub token available: {bool(github_token)}")
        
        # Import the Azure AI Inference SDK
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage as AzureSystemMessage
        from azure.ai.inference.models import UserMessage as AzureUserMessage
        from azure.ai.inference.models import AssistantMessage as AzureAssistantMessage
        from azure.core.credentials import AzureKeyCredential
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from pydantic import Field, BaseModel
        
        # Create the ChatCompletionsClient
        client = ChatCompletionsClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(github_token),
        )
        
        # Test the connection
        print("[DEBUG get_llm] Testing connection with ChatCompletionsClient...")
        try:
            test_response = client.complete(
                messages=[
                    AzureSystemMessage("You are a helpful assistant."),
                    AzureUserMessage("Say 'Connection successful' if you can read this."),
                ],
                temperature=0.7,
                max_tokens=4096,
                model="Meta-Llama-3.1-405B-Instruct"
            )
            print(f"[DEBUG get_llm] Test response: {test_response.choices[0].message.content}")
            print("[DEBUG get_llm] Connection to Azure AI successful!")
        except Exception as e:
            print(f"[DEBUG get_llm] Test connection failed: {str(e)}")
            raise e
        
        # Create a LangChain wrapper for the Azure AI client
        class AzureLlamaWrapper(BaseChatModel, BaseModel):
            """Wrapper for Azure AI Inference ChatCompletionsClient."""
            
            client: Any = Field(description="Azure AI Inference ChatCompletionsClient")
            model_name: str = Field(description="Name of the model to use")
            temperature: float = Field(default=0.2, description="Temperature for sampling")
            max_tokens: int = Field(default=4000, description="Maximum number of tokens to generate")
            
            class Config:
                arbitrary_types_allowed = True
            
            def _generate(self, messages, stop=None, run_id=None, **kwargs):
                """Generate a chat response."""
                # Convert LangChain messages to Azure AI messages
                azure_messages = []
                
                for message in messages:
                    if isinstance(message, SystemMessage):
                        azure_messages.append(AzureSystemMessage(message.content))
                    elif isinstance(message, HumanMessage):
                        azure_messages.append(AzureUserMessage(message.content))
                    elif isinstance(message, AIMessage):
                        azure_messages.append(AzureAssistantMessage(message.content))
                
                print(f"[DEBUG AzureLlamaWrapper] Sending {len(azure_messages)} messages to Azure AI")
                
                # Call the Azure AI Inference API
                response = self.client.complete(
                    messages=azure_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    model=self.model_name
                )
                
                # Extract the response content
                content = response.choices[0].message.content
                print(f"[DEBUG AzureLlamaWrapper] Received response: {content[:100]}...")
                
                message = AIMessage(content=content)
                return ChatResult(generations=[ChatGeneration(message=message)])
            
            @property
            def _llm_type(self):
                """Return the type of LLM."""
                return "azure-llama"
            
            def _call(self, messages, stop=None, run_id=None, **kwargs):
                """Call the LLM with the given messages."""
                result = self._generate(messages, stop, run_id, **kwargs)
                return result.generations[0].message.content
        
        # Create the LLM instance
        print("[DEBUG get_llm] Creating AzureLlamaWrapper instance...")
        llm = AzureLlamaWrapper(
            client=client,
            model_name="Meta-Llama-3.1-405B-Instruct",
            temperature=0.2,
            max_tokens=4000
        )
        
        print("[DEBUG get_llm] Successfully created LLM instance")
        return llm
        
    except Exception as e:
        print(f"[DEBUG get_llm] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # If we get here, we couldn't initialize the Azure LLM, so create a dummy LLM
        print("[DEBUG get_llm] Creating a dummy LLM...")
        from langchain_core.language_models.chat_models import SimpleChatModel
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from pydantic import BaseModel, Field
        
        class DummyLLM(SimpleChatModel, BaseModel):
            """A dummy LLM for testing purposes."""
            
            class Config:
                arbitrary_types_allowed = True
            
            def _generate(self, messages, stop=None, run_id=None, **kwargs):
                message = AIMessage(content="This is a dummy response from the DummyLLM.")
                return ChatResult(generations=[ChatGeneration(message=message)])
            
            @property
            def _llm_type(self):
                return "dummy"
            
            def _call(self, messages, stop=None, run_id=None, **kwargs):
                # Get the last user message for context
                last_user_msg = None
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        last_user_msg = msg.content
                        break
                
                if last_user_msg:
                    # Generate a dummy response that includes a tool call
                    # Use raw string to avoid string formatting issues
                    tool_call = r"""<tool>
{
  "tool_name": "search_knowledge_base",
  "parameters": {
    "query": """ + f'"{last_user_msg[:100]}"' + r"""
  }
}
</tool>"""
                    return f"I'll help you with that. Let me search for information about '{last_user_msg[:50]}'. {tool_call}"
                else:
                    return "I'm a dummy LLM. How can I help you today?"
        
        return DummyLLM()


# -------------------------------------------------------------------------
# Tool Definitions (with docstrings to satisfy the decorator requirement)
# -------------------------------------------------------------------------
@tool
def search_knowledge_base(query: str, search_type: str = "all") -> str:
    """
    Search the user's personal knowledge base for information related to 'query'.
    """
    try:
        # Handle the config parameter that LangChain expects
        return knowledge_base_search_tool(query, search_type=search_type, config={})
    except TypeError:
        # If the function doesn't accept config, try without it
        return knowledge_base_search_tool(query, search_type=search_type)


@tool
def search_documents(query: str) -> str:
    """
    Search uploaded documents for information matching 'query'.
    """
    try:
        # Handle the config parameter that LangChain expects
        return kb_search_documents(query, config={})
    except TypeError:
        # If the function doesn't accept config, try without it
        return kb_search_documents(query)


@tool
def search_artifacts(query: str) -> str:
    """
    Search knowledge artifacts for information matching 'query'.
    """
    try:
        # Handle the config parameter that LangChain expects
        return kb_search_artifacts(query, config={})
    except TypeError:
        # If the function doesn't accept config, try without it
        return kb_search_artifacts(query)


@tool
def search_web(query: str, num_results: int = 3) -> str:
    """
    Search the web for information related to 'query'.
    Args:
        query: The search query string
        num_results: The number of results to retrieve
    """
    try:
        # Handle the config parameter that LangChain expects
        return web_search_tool(query, num_results, config={})
    except TypeError:
        # If the function doesn't accept config, try without it
        return web_search_tool(query, num_results)


@tool
def summarize_text(text: str) -> str:
    """
    Summarize a long piece of text into a concise form.

    Args:
        text: The text to summarize
    Returns:
        A concise summary of the text
    """
    print(f"[DEBUG] summarize_text called with text[:50]: {text[:50]}...")
    try:
        llm = get_llm()
        response = llm.invoke(f"Please summarize the following text concisely:\n\n{text}")
        print(f"[DEBUG] Summarize LLM response: {response.content[:70]}...")
        return response.content
    except Exception as e:
        return f"Error summarizing text: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression (e.g. "2+2", "sin(30)", "sqrt(16)").
    Returns the result as a string.
    """
    import re
    import math
    if re.search(r'[^0-9\s\+\-\*\/\(\)\.\,\^\%\=a-zA-Z]', expression):
        return "Error: Invalid characters in expression."
    expression = expression.replace('^', '**')
    safe_dict = {
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
        'pi': math.pi, 'e': math.e
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
    """
    List all currently available knowledge artifacts.
    """
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
            return f"Title: {artifact['title']}\nAuthor: {artifact['author']}\n\nContent:\n{artifact['content']}"
    except ValueError:
        pass
    for aid, artifact in artifacts.items():
        if artifact_name.lower() in artifact['title'].lower():
            return f"Title: {artifact['title']}\nAuthor: {artifact['author']}\n\nContent:\n{artifact['content']}"
    return f"No artifact found with name or ID: {artifact_name}"


# -------------------------------------------------------------------------
# Tools selection logic
# -------------------------------------------------------------------------
def get_tools(agent_type="wise"):
    """
    Return the appropriate list of tool functions based on 'agent_type'.
    """
    common_tools = [search_knowledge_base, search_web, summarize_text, calculate]
    if agent_type == "scribe":
        return common_tools + [search_documents, search_artifacts, list_artifacts, get_artifact]
    else:
        return common_tools


# -------------------------------------------------------------------------
# Greeting Detector
# -------------------------------------------------------------------------
def handle_greeting(query: str) -> Optional[str]:
    """
    If 'query' is a basic greeting (hi, hello, etc.), return a greeting response;
    otherwise return None.
    """
    # Clean the query by removing punctuation and converting to lowercase
    clean_query = query.lower().strip().rstrip('!?.,:;')
    
    # List of common greetings
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "hola"]
    
    # Check if the entire query is a greeting
    if clean_query in greetings:
        return "Hello! I'm Alexandria, your knowledge assistant. How can I help you today?"
    
    # Check if the query starts with a greeting
    for greeting in greetings:
        if clean_query.startswith(greeting):
            remaining = clean_query[len(greeting):].strip()
            # Check if the rest of the query is empty or contains common follow-ups
            if not remaining or remaining in ["there", "alexandria", "everyone", "all", "friend", "friends"]:
                return "Hello! I'm Alexandria, your knowledge assistant. How can I help you today?"
    
    # Check for common greeting phrases
    common_phrases = ["how are you", "how's it going", "what's up", "how do you do", "nice to meet you"]
    for phrase in common_phrases:
        if phrase in clean_query:
            return "Hello! I'm doing well, thank you for asking. I'm Alexandria, your knowledge assistant. How can I help you today?"
    
    return None


# -------------------------------------------------------------------------
# Create the agent chain with LLM and prompts
# -------------------------------------------------------------------------
def create_agent(agent_type="wise"):
    """
    Create a LangChain pipeline from the system prompts + user messages to the LLM.
    """
    global current_agent_type, current_prompt_template
    
    current_agent_type = agent_type
    if agent_type == "scribe":
        current_prompt_template = DEFAULT_SCRIBE_PROMPT
    else:
        current_prompt_template = DEFAULT_WISE_PROMPT
    
    print(f"[DEBUG create_agent] Using prompt template for agent_type={agent_type}")
    
    try:
        llm = get_llm()
        tools = get_tools(agent_type)
        
        # Format tools for the prompt
        tool_strings = []
        for tool in tools:
            tool_strings.append(f"tool_name: {tool.name}\ndescription: {tool.description}\nparameters: {tool.args}")
        
        formatted_tools = "\n\n".join(tool_strings)
        
        # Create separate system messages instead of a template with placeholders
        system_messages = [
            SystemMessage(content=current_prompt_template),
            SystemMessage(content=f"Available tools:\n\n{formatted_tools}"),
            SystemMessage(content="""
IMPORTANT INSTRUCTIONS:
1. ALWAYS provide a substantive response to the user's query.
2. If you need to use tools, format your tool calls as follows:
   <tool>
   {
     "tool_name": "name_of_the_tool", 
     "parameters": {
       "param1": "value1"
     }
   }
   </tool>
3. NEVER respond with just a tool call - always include explanatory text.
4. If you don't know the answer, say so clearly rather than providing incorrect information.
5. Your response should be at least 2-3 sentences long for most queries.
6. If you're unsure about something, use a tool to look it up rather than guessing.
""")
        ]
        
        print("[DEBUG create_agent] Created system messages")
        
        # Create a function that wraps the LLM call
        def chain(input_dict):
            # Extract the messages
            input_messages = input_dict.get("messages", [])
            
            # Combine system messages with input messages
            all_messages = system_messages + input_messages
            
            # Call the LLM
            return llm.invoke(all_messages)
        
        return chain, tools, None
    except Exception as e:
        print(f"[DEBUG create_agent] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a fixed response function as a fallback
        def fixed_response(_):
            return AIMessage(content="I'm having trouble connecting to my knowledge sources right now. Please try again later or ask a different question.")
        
        return fixed_response, [], None


# -------------------------------------------------------------------------
# Define the custom_tool_node function
# -------------------------------------------------------------------------
def custom_tool_node(state: AgentState) -> dict:
    """
    Finds <tool>...</tool> blocks in the last AI message, executes them, 
    and replaces them with <tool_result>... before handing control back to the agent.
    """
    messages = state["messages"]
    print(f"[DEBUG custom_tool_node] Processing {len(messages)} messages...")
    if not messages or not isinstance(messages[-1], AIMessage):
        print("[DEBUG custom_tool_node] No AIMessage to process for tools. next=END.")
        return {"messages": messages, "next": END}
    
    last_message = messages[-1].content
    print(f"[DEBUG custom_tool_node] Last AIMessage content[:100]: {last_message[:100]!r}")
    
    tool_pattern = r'<tool>\s*({.*?})\s*</tool>'
    tool_matches = re.findall(tool_pattern, last_message, re.DOTALL)
    
    if not tool_matches:
        print("[DEBUG custom_tool_node] No <tool> blocks found. next=END.")
        return {"messages": messages, "next": END}
    
    print(f"[DEBUG custom_tool_node] Found {len(tool_matches)} tool blocks.")
    tools_executed = False
    
    new_content = last_message
    for tool_json in tool_matches:
        try:
            # Try to parse the JSON
            try:
                tool_call = json.loads(tool_json)
            except json.JSONDecodeError as e:
                print(f"[DEBUG custom_tool_node] Error parsing tool call JSON: {e}")
                # Try to fix common JSON issues
                fixed_json = tool_json.replace("'", '"')
                try:
                    tool_call = json.loads(fixed_json)
                except:
                    # If still can't parse, skip this tool call
                    error_msg = f"<tool_result>Error: Could not parse tool call JSON: {str(e)}</tool_result>"
                    new_content = new_content.replace(f"<tool>{tool_json}</tool>", error_msg)
                    continue
            
            tool_name = tool_call.get("tool_name")
            parameters = tool_call.get("parameters", {})
            print(f"[DEBUG custom_tool_node] Attempting tool={tool_name} with params={parameters}")
            
            tools = get_tools(current_agent_type)
            tool_dict = {t.name: t for t in tools}
            
            if tool_name not in tool_dict:
                print(f"[DEBUG custom_tool_node] ERROR: No tool named {tool_name}")
                available_tools = ", ".join(tool_dict.keys())
                error_msg = f"<tool_result>Tool '{tool_name}' not found. Available tools: {available_tools}</tool_result>"
                new_content = new_content.replace(f"<tool>{tool_json}</tool>", error_msg)
                continue
            
            tool_fn = tool_dict[tool_name]
            
            # Handle the special case for search_knowledge_base
            if tool_name == "search_knowledge_base" and "search_type" not in parameters:
                parameters["search_type"] = "all"
            
            # Add a dummy 'config' parameter if the error suggests it's needed
            try:
                # Execute the tool
                if hasattr(tool_fn, '_run'):
                    # For LangChain tools, add a dummy config parameter
                    if tool_name in ["search_knowledge_base", "search_web", "search_documents", "search_artifacts"]:
                        # Try with config parameter added
                        result = tool_fn._run(config={}, **parameters)
                    else:
                        # Regular tools
                        result = tool_fn._run(**parameters)
                else:
                    # For custom tools
                    result = tool_fn(**parameters)
                
                print(f"[DEBUG custom_tool_node] Tool result: {str(result)[:100]}...")
                tools_executed = True
                
                replacement = f"<tool_result>\n{result}\n</tool_result>"
                new_content = new_content.replace(f"<tool>{tool_json}</tool>", replacement)
                
            except TypeError as e:
                if "missing 1 required keyword-only argument: 'config'" in str(e):
                    print(f"[DEBUG custom_tool_node] Adding config parameter for tool '{tool_name}'")
                    try:
                        # Try again with a config parameter
                        result = tool_fn._run(config={}, **parameters)
                        print(f"[DEBUG custom_tool_node] Tool result with config: {str(result)[:100]}...")
                        tools_executed = True
                        
                        replacement = f"<tool_result>\n{result}\n</tool_result>"
                        new_content = new_content.replace(f"<tool>{tool_json}</tool>", replacement)
                    except Exception as inner_e:
                        print(f"[DEBUG custom_tool_node] Error executing tool '{tool_name}' with config: {str(inner_e)}")
                        error_msg = f"<tool_result>Error executing tool '{tool_name}': {str(inner_e)}</tool_result>"
                        new_content = new_content.replace(f"<tool>{tool_json}</tool>", error_msg)
                else:
                    print(f"[DEBUG custom_tool_node] Error executing tool '{tool_name}': {str(e)}")
                    error_msg = f"<tool_result>Error executing tool '{tool_name}': {str(e)}</tool_result>"
                    new_content = new_content.replace(f"<tool>{tool_json}</tool>", error_msg)
            except Exception as e:
                print(f"[DEBUG custom_tool_node] Error executing tool '{tool_name}': {str(e)}")
                error_msg = f"<tool_result>Error executing tool '{tool_name}': {str(e)}</tool_result>"
                new_content = new_content.replace(f"<tool>{tool_json}</tool>", error_msg)
            
        except Exception as e:
            print(f"[DEBUG custom_tool_node] Exception: {e}")
            import traceback
            traceback.print_exc()
            error_msg = f"<tool_result>Error processing tool call: {str(e)}</tool_result>"
            new_content = new_content.replace(f"<tool>{tool_json}</tool>", error_msg)
    
    # Create a response that informs the agent about the maximum iterations
    new_message = AIMessage(content=new_content)
    
    # Add a system message telling the agent to analyze <tool_result> sections
    system_message = SystemMessage(content=(
        "Tool execution complete. Please review <tool_result> content. "
        "If relevant, integrate it into your final answer. "
        "If nothing was found, please answer with what you already know without using tools again."
    ))
    
    if not tools_executed:
        print("[DEBUG custom_tool_node] No tools executed successfully. We'll instruct the agent to answer directly.")
        direct_answer_message = SystemMessage(
            content="No tools were successfully executed. Please answer the user's question directly using your knowledge without trying to use tools again."
        )
        return {"messages": messages[:-1] + [direct_answer_message], "next": "agent_executor"}
    
    return {"messages": messages[:-1] + [new_message, system_message], "next": "agent_executor"}


# -------------------------------------------------------------------------
# Define the agent_executor function
# -------------------------------------------------------------------------
def agent_executor(state):
    """
    Execute the agent workflow.
    """
    messages = state["messages"]
    print(f"[DEBUG agent_executor] Processing {len(messages)} messages...")
    
    # Get the last user message for debugging
    last_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break
    
    if last_user_msg:
        print(f"[DEBUG agent_executor] Last user message: {last_user_msg[:100]}...")
    
    # Check if we've exceeded the maximum number of iterations
    # Count how many times we've been through this node with the same user message
    iteration_count = 0
    for msg in messages:
        if isinstance(msg, SystemMessage) and "Iteration count:" in msg.content:
            try:
                iteration_count = int(msg.content.split("Iteration count:")[1].strip())
                break
            except:
                pass
    
    # If we've exceeded the maximum number of iterations, return a final response
    max_iterations = 3  # Maximum number of iterations to prevent infinite loops
    if iteration_count >= max_iterations:
        print(f"[DEBUG agent_executor] Exceeded maximum iterations ({max_iterations})")
        error_message = AIMessage(content=f"I understand you're asking about something, but I'm having trouble processing it right now. Could you please rephrase your question or try a different topic?")
        return {
            "messages": messages + [error_message],
            "next": END
        }
    
    # Increment the iteration count
    iteration_count += 1
    iteration_message = SystemMessage(content=f"Iteration count: {iteration_count}")
    messages = [msg for msg in messages if not (isinstance(msg, SystemMessage) and "Iteration count:" in msg.content)]
    messages.append(iteration_message)
    
    try:
        # Get the agent chain and tools
        agent_chain, tools, _ = create_agent(current_agent_type)
        
        try:
            # Invoke the agent chain with the messages
            response = agent_chain({"messages": messages})
            print(f"[DEBUG agent_executor] LLM response: {response.content[:200]}...")
            
            # Check if the response is empty or too short
            if not response.content or len(response.content.strip()) < 10:
                print("[DEBUG agent_executor] Response too short, forcing tool call")
                if last_user_msg and len(last_user_msg) > 10:
                    # Use raw string to avoid string formatting issues
                    tool_call = r"""<tool>
{
  "tool_name": "search_knowledge_base",
  "parameters": {
    "query": """ + f'"{last_user_msg[:100]}"' + r"""
  }
}
</tool>"""
                    response = AIMessage(content=f"Let me search for information about that. {tool_call}")
        
        except Exception as invoke_error:
            print(f"[DEBUG agent_executor] Error invoking LLM: {str(invoke_error)}")
            # If there's an error with the formatted prompt, try a simpler approach
            response = AIMessage(content=f"I'm having trouble understanding your request. Could you please provide more details or rephrase your question?")
        
        # Add the response to the messages
        messages.append(response)
        
        # Return the updated state
        return {
            "messages": messages,
            "next": "custom_tool_node"  # Always go to the tool node next
        }
    except Exception as e:
        print(f"[DEBUG agent_executor] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return an error message
        error_message = AIMessage(content="I'm having technical difficulties right now. Please try again later.")
        return {
            "messages": messages + [error_message],
            "next": END
        }


# Helper function for formatting tools
def format_tools_for_prompt(tools):
    """Format tools for the prompt."""
    tool_strings = []
    for tool in tools:
        tool_strings.append(f"tool_name: {tool.name}\ndescription: {tool.description}\nparameters: {tool.args}")
    return "\n\n".join(tool_strings)


# -------------------------------------------------------------------------
# Build & Compile the LangGraph
# -------------------------------------------------------------------------
def build_graph(agent_type="wise"):
    """
    Build the agent graph using LangGraph, properly connecting the LLM, tools, and workflow.
    Following the migration guide: https://python.langchain.com/docs/how_to/migrate_agent/
    """
    print(f"[DEBUG build_graph] Building graph for agent_type={agent_type}")
    global current_agent_type
    current_agent_type = agent_type
    
    # Get the LLM and tools
    llm = get_llm()
    tools_list = get_tools(agent_type)
    
    # Create a prompt that properly formats tool instructions
    if agent_type == "scribe":
        system_prompt = DEFAULT_SCRIBE_PROMPT
    else:
        system_prompt = DEFAULT_WISE_PROMPT
    
    # Format the tools for the prompt
    tool_strings = []
    for tool in tools_list:
        tool_strings.append(f"tool_name: {tool.name}\ndescription: {tool.description}\nparameters: {tool.args}")
    
    formatted_tools = "\n\n".join(tool_strings)
    
    # Create the prompt template WITHOUT using string formatting in the template itself
    # The tool instructions are directly included in the system message
    tool_instructions = """
IMPORTANT: You have access to the following tools described above.

To use a tool, use the following format:
<tool>
tool_json_object
</tool>

Where tool_json_object is a JSON object with the following structure:
{
  "tool_name": "name_of_the_tool",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}

The tool will respond with:
<tool_result>
result from the tool
</tool_result>

Always think step-by-step about what the user is asking. If you need to use a tool, use it.
If you don't need to use a tool, just respond directly.

DO NOT REPEAT THE USER'S QUESTION in your response.
DO NOT START YOUR RESPONSE WITH GREETINGS like "Hello" or "Hi there".
DO NOT GENERATE MULTIPLE RESPONSES in a single turn.

If a tool fails, try once more with a different approach or answer directly from your knowledge.
"""
    
    # Create a message templates list with a template for formatted_tools
    messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=f"Available tools:\n\n{formatted_tools}"),
        SystemMessage(content=tool_instructions)
    ]
    
    # Create the workflow with a higher recursion limit
    # Set the recursion limit correctly as a config parameter
    from langgraph.graph import Settings
    config = Settings(recursion_limit=50)
    workflow = StateGraph(AgentState, config=config)
    
    # Define the agent executor function
    def agent_executor(state):
        """Execute the agent with the current state."""
        state_messages = state["messages"]
        
        print(f"[DEBUG agent_executor] Processing {len(state_messages)} messages")
        
        # Check for iteration count to prevent infinite loops
        iteration_count = 0
        for msg in state_messages:
            if isinstance(msg, SystemMessage) and msg.content.startswith("Iteration count:"):
                try:
                    iteration_count = int(msg.content.split(":", 1)[1].strip())
                    break
                except (ValueError, IndexError):
                    pass
        
        # If we've already tried multiple times, skip tools and answer directly
        max_iterations = 3
        if iteration_count >= max_iterations:
            print(f"[DEBUG agent_executor] Exceeded max iterations ({max_iterations}), forcing direct answer")
            direct_answer_message = SystemMessage(content=(
                f"You have attempted to use tools {iteration_count} times without success. "
                "Please answer the user's question directly without using any tools. "
                "Use your own knowledge to provide the best possible answer."
            ))
            state_messages = [msg for msg in state_messages if not msg.content.startswith("Iteration count:")]
            state_messages.append(direct_answer_message)
        
        # Increment iteration count for next time
        state_messages = [msg for msg in state_messages if not msg.content.startswith("Iteration count:")]
        state_messages.append(SystemMessage(content=f"Iteration count: {iteration_count + 1}"))
        
        # Combine the system messages with user messages
        all_messages = messages + state_messages
        
        # Get the response from the LLM
        ai_message = llm.invoke(all_messages)
        
        # Add the AI message to the state
        state_messages.append(ai_message)
        
        # If we hit the max iterations, force ending the conversation
        if iteration_count >= max_iterations:
            print("[DEBUG agent_executor] Max iterations reached, ending conversation")
            return {"messages": state_messages, "next": END}
        
        # Check if the message contains a tool call
        if "<tool>" in ai_message.content:
            print("[DEBUG agent_executor] Tool call detected, routing to custom_tool_node")
            return {"messages": state_messages, "next": "custom_tool_node"}
        else:
            print("[DEBUG agent_executor] No tool call detected, ending workflow")
            return {"messages": state_messages, "next": END}
    
    # Define the entry point function
    def entry_point_fn(state):
        """Entry point for the workflow that processes messages."""
        messages = state["messages"]
        
        if not messages:
            return {"messages": messages, "next": END}
        
        print(f"[DEBUG entry_point_fn] Processing {len(messages)} messages")
        
        # Always route to the agent_executor
        return {"messages": messages, "next": "agent_executor"}
    
    # Add nodes to the workflow
    workflow.add_node("entry_point", entry_point_fn)
    workflow.add_node("agent_executor", agent_executor)
    workflow.add_node("custom_tool_node", custom_tool_node)
    
    # Add edges
    workflow.add_edge("entry_point", "agent_executor")
    workflow.add_edge("agent_executor", "custom_tool_node")
    workflow.add_edge("custom_tool_node", "agent_executor")
    workflow.add_edge("agent_executor", END)
    workflow.add_edge("custom_tool_node", END)
    
    # Set the entry point
    workflow.set_entry_point("entry_point")
    
    # Compile the workflow without directly passing recursion_limit
    return workflow.compile()


# -------------------------------------------------------------------------
# Initialize the global compiled_workflow
# -------------------------------------------------------------------------
compiled_workflow = build_graph("wise")


# -------------------------------------------------------------------------
# Helper to update the agent type & recompile
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# Optional: A direct function to get a single response from the agent
# -------------------------------------------------------------------------
def get_agent_response(message, chat_history, agent_type="wise"):
    """
    Get a response from the agent.
    
    Args:
        message: The message to respond to
        chat_history: The chat history, either as a string or a list of message objects
        agent_type: The type of agent to use (wise or scribe)
        
    Returns:
        A dictionary containing the response
    """
    print(f"[DEBUG get_agent_response] Called with agent_type={agent_type}")
    
    # Get the workflow for the agent type
    workflow = get_workflow(agent_type)
    
    # Prepare messages for the agent state
    messages = []
    
    # Process chat history if it exists
    if chat_history:
        if isinstance(chat_history, str):
            # Process string format chat history (from app.py)
            print("[DEBUG get_agent_response] Processing string chat history")
            lines = chat_history.split('\n')
            
            for line in lines:
                if line.startswith("Human: "):
                    content = line[7:]  # Remove "Human: " prefix
                    if content.strip():
                        messages.append(HumanMessage(content=content))
                elif line.startswith("AI: "):
                    content = line[4:]  # Remove "AI: " prefix
                    if content.strip():
                        messages.append(AIMessage(content=content))
        elif isinstance(chat_history, list):
            # Process list format chat history
            print("[DEBUG get_agent_response] Processing list chat history")
            for msg in chat_history:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user" and content:
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant" and content:
                        messages.append(AIMessage(content=content))
                    elif role == "system" and content:
                        messages.append(SystemMessage(content=content))
                elif isinstance(msg, str) and msg.strip():
                    # Assume it's a user message if it's just a string
                    messages.append(HumanMessage(content=msg))
    
    # Add the current message if it's not empty
    if message and isinstance(message, str) and message.strip():
        print(f"[DEBUG get_agent_response] Adding current message: {message[:30]}...")
        messages.append(HumanMessage(content=message))
    
    # If no messages were processed, create a default one
    if not messages:
        print("[DEBUG get_agent_response] No valid messages found, using default")
        messages.append(HumanMessage(content="Hello"))
    
    # Check if the last message is a greeting
    if messages and isinstance(messages[-1], HumanMessage):
        greeting_response = handle_greeting(messages[-1].content)
        if greeting_response:
            print("[DEBUG get_agent_response] Detected greeting, returning fixed response")
            return {"response": greeting_response}
    
    # Create the agent state
    state = {"messages": messages}
    
    # Invoke the workflow with the state
    print(f"[DEBUG get_agent_response] Invoking workflow with {len(messages)} messages")
    try:
        # Invoke the workflow
        result = workflow.invoke(state)
        
        # Get the last message from the result
        if result["messages"] and isinstance(result["messages"][-1], AIMessage):
            response = result["messages"][-1].content
            print("[DEBUG get_agent_response] Got response from workflow")
            return {"response": response}
        else:
            print("[DEBUG get_agent_response] No AI message in result, returning fallback")
            return {"response": generate_fallback_response(messages[-1].content if messages else "")}
    except Exception as e:
        print(f"[DEBUG get_agent_response] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"response": f"I encountered an error while processing your request: {str(e)}. Please try again or contact support if the issue persists."}


# -------------------------------------------------------------------------
# Fallback response generator when LLM fails
# -------------------------------------------------------------------------
def generate_fallback_response(query: str) -> str:
    """
    Generate a simple fallback response when the LLM fails to provide a valid response.
    This doesn't use the LLM, so it should always work.
    """
    print(f"[DEBUG] Generating fallback response for query: {query[:50]}...")
    
    # Check for common query types and provide basic responses
    query_lower = query.lower()
    
    if any(greeting in query_lower for greeting in ["hello", "hi", "hey", "greetings"]):
        return "Hello! I'm Alexandria, your AI assistant. How can I help you today?"
    
    if "how are you" in query_lower:
        return "I'm functioning well, thank you for asking. How can I assist you today?"
    
    if any(word in query_lower for word in ["help", "assist", "support"]):
        return "I'm here to help you with information, answering questions, and assisting with various tasks. What specific help do you need today?"
    
    if any(word in query_lower for word in ["search", "find", "look for"]):
        return "I can search for information in your knowledge base, documents, or on the web. Please try again with a more specific query."
    
    if "thank" in query_lower:
        return "You're welcome! Feel free to ask if you need anything else."
    
    # Default generic response
    return "I understand you're asking about something, but I'm having trouble processing it right now. Could you please rephrase your question or try a different topic?"


def get_workflow(agent_type="wise"):
    """
    Get the workflow for the specified agent type.
    Will build a new workflow if one doesn't exist for this agent type.
    
    Args:
        agent_type: Type of agent to use (wise or scribe)
        
    Returns:
        The workflow for the agent
    """
    # Use a global dictionary to cache workflows
    global _workflows
    if '_workflows' not in globals():
        _workflows = {}
    
    # If we don't have a cached workflow for this agent type, build one
    if agent_type not in _workflows:
        print(f"[DEBUG get_workflow] Building new workflow for agent_type={agent_type}")
        _workflows[agent_type] = build_graph(agent_type)
    
    return _workflows[agent_type]