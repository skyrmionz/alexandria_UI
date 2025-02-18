import os
from flask import Flask, request, jsonify
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from tools.knowledge_base_tool import knowledge_base_search_tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv 

load_dotenv()
pod_id = os.getenv("POD_ID")

app = Flask(__name__)

# Default prompt templates for each agent type:
DEFAULT_WISE_PROMPT = """
You are Alexandria the Wise. You are a chatbot that answers questions for the user. Be friendly and keep your responses to 1000 characters.
You have access to the user's knowledge database, which retrieves information from the document repository (which may include resumes, reports, etc.).
Always use these tools {tools}, specifically [{tool_names}], to help you answer questions.

{input}

Your response should have two sections:
1. <think> ... </think> - a detailed chain-of-thought that is not visible to the user.
2. Final Answer: [Your concise, HTML-stylized answer based solely on retrieved knowledge]

ALWAYS stylize your text with ONLY HTML tags to make it more readable (e.g., <b> bolded text</b>) instead of **text**. Make sure your line spacing is consistent.

IMPORTANT: You only have 30 tries to get to a final answer.

Before producing your final answer, determine whether your answer requires additional information from the document database.
If so, ensure that your answer is directly supported by the retrieved data.

{agent_scratchpad}
"""

DEFAULT_SCRIBE_PROMPT = """
You are Alexandria the Scribe. Your task is to revise and rewrite documents to improve clarity, tone, and structure.
You have access to the user's documents and previous context.
Always produce a revised version that maintains the original meaning but is more clear and engaging.

{input}

Your response should have two sections:
1. <think> ... </think> - an internal chain-of-thought not visible to the user.
2. Final Revised Document: [Your concise, HTML-stylized revised version of the provided text]

Ensure you use ONLY HTML tags for styling (e.g., <b> for bold) and keep consistent line spacing.
You have 30 attempts to produce a final revised document.

{agent_scratchpad}
"""

# Global state: current prompt template and agent type.
current_prompt_template = DEFAULT_WISE_PROMPT
current_agent_type = "wise"  # "wise" or "scribe"

ollama_url = os.getenv('OLLAMA_URL', f'https://{pod_id}.proxy.runpod.net/')

def initialize_agent():
    custom_prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        template=current_prompt_template
    )
    tools = [
        Tool(
            name="KnowledgeBaseSearch",
            func=knowledge_base_search_tool,
            description="Searches the document database for relevant information."
        )
    ]
    llm = ChatOllama(
        model="deepseek-r1:8b",
        base_url=ollama_url,
        temperature=0.6,
        max_tokens=32000,
    )
    memory = ConversationBufferMemory(
        return_messages=True,
        human_prefix="Human:",
        ai_prefix="AI:"
    )
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=custom_prompt,
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=30
    )
    return agent_executor

# Initialize the agent executor once.
agent_executor = initialize_agent()

def update_agent(new_agent_type, new_prompt_template=None):
    """
    Only updates the cosmetic selection by setting the global variables.
    The agent executor will be reinitialized when a chat message is sent.
    """
    global current_agent_type, current_prompt_template
    if new_agent_type not in ["wise", "scribe"]:
        new_agent_type = "wise"
    current_agent_type = new_agent_type
    if new_prompt_template and new_prompt_template.strip():
        current_prompt_template = new_prompt_template
    else:
        current_prompt_template = DEFAULT_WISE_PROMPT if new_agent_type == "wise" else DEFAULT_SCRIBE_PROMPT
    # Note: We do not reinitialize agent_executor here.

def get_agent_response(query: str) -> str:
    # Reinitialize the agent executor with the current prompt template so that the
    # appropriate prompt is used based on the cosmetic selection.
    global agent_executor
    agent_executor = initialize_agent()
    
    kb_context = knowledge_base_search_tool(query)
    combined_input = f"Knowledge Document Database Results: {kb_context}\nQuestion: {query}"
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

@app.route("/chat_api", methods=["POST"])
def chat_api():
    message = request.form.get("message", "")
    response = get_agent_response(message)
    return jsonify({"response": response})

@app.route("/update_agent", methods=["POST"])
def update_agent_route():
    agent_type = request.form.get("agent_type")
    prompt = request.form.get("prompt")
    try:
        update_agent(agent_type, prompt)
        return jsonify({"message": "Agent updated successfully."})
    except Exception as e:
        return jsonify({"message": f"Error updating agent: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)