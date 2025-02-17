import os
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from tools.knowledge_base_tool import knowledge_base_search_tool
from tools.web_search_tool import web_search_tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Retrieve the Ollama URL from environment variables, defaulting to a given address
ollama_url = os.getenv('OLLAMA_URL', 'https://cjalmexcy9f8qs-11434.proxy.runpod.net/')

# Revised prompt template:
# - Removed separate {kb_context} variable.
# - The knowledge base evidence will be included in {input} along with the user's question.
template = """
You are Alexandria. You are a chatbot that answers questions for the user. Be friendly and keep your responses to 1000 characters.
You have access to the user's knowledge database, which retrieves information from the document repository (which may include resumes, reports, etc.).
Always use these tools {tools}, specifically [{tool_names}], to help you answer questions.

{input}

Your response should have two sections:
1. <think> ... </think> - a detailed chain-of-thought that is not visible to the user.
2. Final Answer: [Your concise, HTML-stylized answer based solely on retrieved knowledge]

ALWAYS stylize your text with ONLY HTML tags to make it more readable (e.g., <b> bolded text</b>) instead of **text**.

IMPORTANT: You only have 30 tries to get to a final answer.

Before producing your final answer, determine whether your answer requires additional information from the document database.
If so, ensure that your answer is directly supported by the retrieved data.

Provide plenty of detail in your final answer to support your claims fully.

{agent_scratchpad}
"""

# The ReAct agent automatically supplies "agent_scratchpad", "tools", and "tool_names".
custom_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template=template
)

tools = [
    Tool(
        name="KnowledgeBaseSearch",
        func=knowledge_base_search_tool,
        description="Searches the document database for relevant information."
    )
]

# Instantiate the ChatOllama model
llm = ChatOllama(
    model="deepseek-r1:8b",  # Update if needed
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
    max_iterations=30
)

def get_agent_response(query: str) -> str:
    # Pre-call the KnowledgeBaseSearch tool to get supporting evidence
    kb_context = knowledge_base_search_tool(query)
    
    # Combine the knowledge base evidence and the user's question into one string.
    # This combined string will be used as the sole input.
    combined_input = f"Knowledge Document Database Results: {kb_context}\nQuestion: {query}"
    
    # Pass only the "input" key (the agent will auto-fill "agent_scratchpad", "tools", and "tool_names")
    input_data = {"input": combined_input}
    
    try:
        result = agent_executor.invoke(input_data)
        if 'output' in result:
            return result['output']
        else:
            return "I'm sorry, I wasn't able to generate a response."
    except Exception as e:
        partial = getattr(e, "partial_result", None)
        if partial:
            return f"Partial response: {partial}"
        return f"An error occurred: {str(e)}"