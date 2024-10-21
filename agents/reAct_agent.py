# agent.py

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from tools.search_tool import tools
import os

# Load environment variables
load_dotenv()

# OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model_name="gpt-4o-mini",  # You can use "gpt-4" if you have access
    temperature=0,
    api_key=api_key
)

# Pull the ReAct prompt template from the LangChain hub
prompt = hub.pull("hwchase17/react")

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
    # verbose=True
)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# Function to process user input
def process_user_input(user_input):
    print('🚀 user_input 🚀-->>', user_input)
    response = agent_executor.invoke({"input": user_input})
    print('🚀 response 🚀-->>', response)
    return response

if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        response = process_user_input(user_input)
        print("Assistant:", response)
