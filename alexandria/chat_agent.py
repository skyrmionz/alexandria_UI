import os
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from tools.knowledge_base_tool import knowledge_base_search_tool
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv 

# New imports for persistence using LangGraph checkpoints:
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()
pod_id = os.getenv("POD_ID")

# Default prompt templates for each agent type:
DEFAULT_WISE_PROMPT = """
Your name is Alexandria. Your primary goal is to answer the user's question and complete their requests. Be friendly and excited! Answer in 2000-3000 characters and make sure to give detailed answers with evidence-backed claims from the sources.
You have access to the user's documents and previous conversation history. Make sure to look at the conversation history to see if the user is continuing their query.
You have access to the user's knowledge database, which retrieves information from the document repository.

You can use these tools {tools}, specifically [{tool_names}], to help you answer questions.

Knowledge Document Database Results: {input}

IMPORTANT: If any Knowledge Document Database Results are returned in the input, you MUST incorporate them to ground your final answer.
ONLY use information from the Knowledge Document Database. You don't need to mention the Knowledge Document Database in your response.

Your response should have two sections:
1. <think> ... </think> - an internal chain-of-thought that is not visible to the user.
2. Final Answer: [Your concise, HTML-stylized answer based solely on retrieved knowledge] (Do not apply ANY color to the text and keep line breaks to only one line)

{agent_scratchpad}
"""

DEFAULT_SCRIBE_PROMPT = """
Your name is Alexandria. Be friendly and excited! Your job is to create, revise, and rewrite documents (also called Artifacts) from the Knowledge Document Database to improve clarity, tone, and structure. Answer in 500-5000 characters.
If the user wants to create, revise, or rewrite an artifact, you are responsible for doing it.

You can use these tools {tools}, specifically [{tool_names}], to help you find the correct document to revise.

Knowledge Document Database Results: {input}

You don't need to mention the Knowledge Document Database in your response.

IMPORTANT: Use only the relevant documents provided.
Your response should ALWAYS have these two sections:
1. <think> ... </think> - internal chain-of-thought explaining what changes you're making and why.
2. Final Document: [ALWAYS provide the COMPLETE document with ALL revisions, using proper HTML formatting] - 
You MUST include the entire document word-for-word in the content field, along with the revisions you've made.

When formatting the document, you MUST follow these HTML styling guidelines to match the article view:
- Use <h1> for the main title (REQUIRED)
- Use <h3> for section headers (REQUIRED for any sections)
- Use <p> for paragraphs (REQUIRED for main content)
- Use <b> or <strong> for emphasis on important terms or phrases
- Use <i> or <em> for secondary emphasis
- Use <ul> and <li> for bullet points (REQUIRED for lists)
- Add style="margin-left: 5px;" to list items
- Use <br> for spacing between sections
- Keep consistent spacing with <br> tags between major sections
- Ensure headers are followed by their content without extra spacing

Example document structure:
<h1>Document Title</h1>
<p id="article-author">Author: [Author Name]</p>
<br>
<h3>Section Title</h3>
<p>Section content with <b>important text</b> and <i>emphasized points</i>.</p>
<br>
<h3>Bullet Points</h3>
<ul>
    <li style="margin-left: 5px;">First point with detail</li>
    <li style="margin-left: 5px;">Second point with detail</li>
</ul>

IMPORTANT: When the user says "Create an artifact from this revision" or similar, you MUST:
1. Look at your MOST RECENT response only (ignore older messages)
2. Extract a meaningful title from the content
3. Use the appropriate author information if available, or "Alexandria" if not
4. Format your response EXACTLY like this, including ALL the content:

<think>
Creating an artifact from my last revision. I'll format it properly with a title and complete content.
</think>

<create_artifact>
{
    "title": "[Extracted title]",
    "author": "[Author name]",
    "content": "[COMPLETE HTML-formatted document with ALL content and styling]"
}
</create_artifact>

✨ Perfect! I've created a new artifact for you. You can find it in the Knowledge Artifacts section.

IMPORTANT REMINDERS:
- ALWAYS include the complete document content, not just sections or summaries
- ALWAYS use proper HTML formatting for the entire document
- ALWAYS include both the <think> section and the complete document in your responses
- When creating an artifact, ALWAYS include the entire document in the content field

{agent_scratchpad}
"""

# Global state: current prompt template and agent type.
current_prompt_template = DEFAULT_WISE_PROMPT
current_agent_type = "wise"  # "wise" or "scribe"
ollama_url = os.getenv('OLLAMA_URL', f'https://{pod_id}.proxy.runpod.net/')

def update_agent(new_agent_type, new_prompt_template=None):
    """
    Updates the agent type and prompt template, and reinitializes the agent executor.
    """
    global current_agent_type, current_prompt_template, agent_executor
    if new_agent_type not in ["wise", "scribe"]:
        new_agent_type = "wise"
    current_agent_type = new_agent_type
    if new_prompt_template and new_prompt_template.strip():
        current_prompt_template = new_prompt_template
    else:
        current_prompt_template = DEFAULT_WISE_PROMPT if new_agent_type == "wise" else DEFAULT_SCRIBE_PROMPT
    
    # Reinitialize the agent executor with the new prompt
    agent_executor = initialize_agent()
    print(f"Agent updated to type: {current_agent_type}")
    return agent_executor

def initialize_agent():
    # Use persistent memory from LangGraph.
    memory_saver = MemorySaver()
    
    # Build prompt template by prepending conversation history.
    combined_template = "Conversation History:\n{chat_history}\n\n" + current_prompt_template
    custom_prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names", "chat_history"],
        template=combined_template
    )
    tools = [
        Tool(
            name="KnowledgeBaseSearch",
            func=knowledge_base_search_tool,
            description="Searches the document database for relevant information."
        )
    ]
    llm = ChatOllama(
        model="llama3.1:8b",
        base_url=ollama_url,
        temperature=0.6,
        max_tokens=32000,
    )
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=custom_prompt,
    )
    # Do not pass memory here to AgentExecutor; we'll handle conversation via the workflow.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        max_execution_time=120,
    )
    return agent_executor

# Initialize the agent executor once.
agent_executor = initialize_agent()

# Build a workflow for persistent conversation history using LangGraph.
workflow = StateGraph(state_schema=MessagesState)

def call_agent(state: MessagesState):
    # Ensure messages key exists.
    if "messages" not in state:
        state["messages"] = []
    # Build conversation history string from state messages.
    chat_history = "\n".join([msg.content for msg in state["messages"]])
    # Use the latest human message for context.
    query = state["messages"][-1].content if state["messages"] else ""
    # Call the agent with additional variables.
    response = get_agent_response(query, chat_history)
    state["messages"].append(AIMessage(content=response))
    return {"messages": state["messages"]}

workflow.add_node("agent", call_agent)
workflow.add_edge(START, "agent")

# Use MemorySaver as our checkpointer.
memory_saver = MemorySaver()
compiled_workflow = workflow.compile(checkpointer=memory_saver)

def get_agent_response(query: str, chat_history: str) -> str:
    try:
        # More flexible greeting detection
        query_lower = query.lower().strip()
        greeting_words = {"hello", "hi", "hey", "greetings"}
        common_additions = {"there", "alexandria", "how are you", "how's it going"}
        
        # Check if the query is a simple greeting
        words = set(query_lower.replace("!", "").replace("?", "").split())
        if (any(word in greeting_words for word in words) and 
            words.issubset(greeting_words | common_additions)):
            return "Hello! I'm Alexandria, your friendly AI assistant. How can I help you today?"
        
        # For all other queries, provide thorough responses
        kb_context = knowledge_base_search_tool(query)
        combined_input = f"Knowledge Document Database Results: {kb_context}\nQuestion: {query}"
        input_data = {
            "input": combined_input,
            "chat_history": chat_history,
            "agent_scratchpad": "",
            "tools": "KnowledgeBaseSearch",
            "tool_names": "KnowledgeBaseSearch"
        }
        
        result = agent_executor.invoke(input_data)
        if 'output' in result:
            return result['output']
        return "I'm sorry, I wasn't able to generate a response."
        
    except Exception as e:
        print(f"Error in get_agent_response: {str(e)}")
        partial = getattr(e, "partial_result", None)
        if partial:
            return f"Partial response: {partial}"
        return f"An error occurred while processing your request: {str(e)}"