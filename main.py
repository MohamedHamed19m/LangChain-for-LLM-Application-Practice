from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

# So if you run out of requests on 2.5 Flash
#  you can still switch to 2.5 Flash-Lite (1,000/day!) or 2.0 Flash-Lite, and keep going — as long as you stay within each model’s limit.

load_dotenv()

# Define tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except Exception:
        return "Error in calculation"

tools = [calculator]


# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    convert_system_message_to_human=True,
)

# Add memory via LangGraph persistence
checkpointer = MemorySaver()
agent_executor = create_react_agent(llm, tools, checkpointer=checkpointer)

# Each conversation needs a thread_id (like a session ID)
thread = {"configurable": {"thread_id": "user-123"}}

# Conversation loop
for user_msg in [
    "Hello, who are you?",
    "Can you calculate 12 * 8 + 5?",
    "What did I ask before the math question?",
]:
    result = agent_executor.invoke({"messages": [("user", user_msg)]}, config=thread)
    print(result["messages"][-1].content)
