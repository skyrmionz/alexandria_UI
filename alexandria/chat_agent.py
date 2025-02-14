# chat_agent.py
import os
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from tools.knowledge_base_tool import knowledge_base_search_tool
from tools.web_search_tool import web_search_tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Retrieve the Ollama URL from environment variables
ollama_url = os.getenv('OLLAMA_URL', 'http://ollama:11434')

# Define a simpler prompt that explicitly instructs to keep steps under 10
template = """
You are Alexandria. Be friendly, concise, and do not overthink your answers.
You have access to the following tools:

{tools}

Use the following format:

Thought Process #: This is the number of the thought process attempt you're on. You have 10 to reach a Final Answer.
Query: the user's question or query you must address
Thought: you should always think about what to do
Action: the action to take: you can use one or none of these: [{tool_names}]
Action Result: You know what to do now.
Thought: You now know the final answer
Final Answer: This is your final answer to the original question

IMPORTANT: You have 10 thought processes to reach a final answer.

Begin!

Question: {input}
{agent_scratchpad}
"""

custom_prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    template=template
)

# Define your tools
tools = [
    Tool(
        name="KnowledgeBaseSearch",
        func=knowledge_base_search_tool,
        description="Searches the local knowledge base for relevant information."
    ),
    Tool(
        name="WebSearch",
        func=web_search_tool,
        description="Performs a web search using DuckDuckGo."
    )
]

# Instantiate the ChatOllama model
llm = ChatOllama(
    model="deepseek-r1:latest",  # Update if needed
    base_url=ollama_url,
    temperature=0.7,
    max_tokens=1500
)

# Create memory for conversation context
memory = ConversationBufferMemory(
    return_messages=True,
    human_prefix="Human:",
    ai_prefix="AI:"
)

# Create the ReACT-style agent with our custom prompt
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=custom_prompt,
)

# Create the AgentExecutor with fewer max_iterations
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10         # Try to wrap up in 10 steps
)

def get_agent_response(query: str) -> str:
    try:
        result = agent_executor.invoke({"input": query})
        if 'output' in result:
            return result['output']
        else:
            return "I'm sorry, I wasn't able to generate a response."
    except Exception as e:
        partial = getattr(e, "partial_result", None)
        if partial:
            return f"Partial response: {partial}"
        return f"An error occurred: {str(e)}"