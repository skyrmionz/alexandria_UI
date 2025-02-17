import os
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from tools.knowledge_base_tool import knowledge_base_search_tool
from tools.web_search_tool import web_search_tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Retrieve the Ollama URL from environment variables, defaulting to the new RunPod DNS address
ollama_url = os.getenv('OLLAMA_URL', 'https://tepm2e161hnrjt-11434.proxy.runpod.net/')

# Define a simpler prompt that explicitly instructs to keep steps under 10
template = """
You are Alexandria. Be friendly and keep your responses to 1000 characters.
You have access to the following tools, which grabs information from the knowledge base:

{tools}

Your response should have two sections:
1. <think> ... </think> - a detailed chain-of-thought that is not visible to the user.
2. Final Answer: [Your concise, HTML-stylized answer based solely on retrieved knowledge]

ALWAYS use these actions: [{tool_names}] - ONLY use information found in the knowledge base for your answer
ALWAYS stylize your text with HTML tags to make it more readable. (Example: <b> bolded text</b>)

IMPORTANT: You only have 10 tries to get to a final answer. 

Before producing your final answer, ensure that the information is directly supported by the retrieved knowledge base data.

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
        description="Searches the knowledge base for relevant information to ground your response."
    ),
    #Tool(
    #    name="WebSearch",
    #    func=web_search_tool,
    #    description="Performs a web search using DuckDuckGo."
    #)
]

# Instantiate the ChatOllama model
llm = ChatOllama(
    model="deepseek-r1:latest",  # Update if needed
    base_url=ollama_url,
    temperature=0.6,
    max_tokens=32000,
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
    max_iterations=10
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