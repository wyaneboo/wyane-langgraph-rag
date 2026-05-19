import os
import sys
from typing import TypedDict, List, Union
from google import genai
from google.genai import errors
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langsmith import wrappers
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

gemini_client = genai.Client()
# Wrap the Gemini client to enable LangSmith tracing
client = wrappers.wrap_gemini(
    gemini_client,
    tracing_extra={
        "tags": ["gemini", "python"],
        "metadata": {
            "integration": "google-genai",
        },
    },
)

MODEL = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    contents = "\n".join(
        f"User: {message.content}" if isinstance(message, HumanMessage)
        else f"AI: {message.content}"
        for message in state["messages"]
    )

    try:
        response = client.models.generate_content(
                    model=MODEL,
                    contents=contents,
        )
    except errors.APIError as exc:
        error_message = f"Google GenAI API error while using {MODEL}: {exc}"
        print(f"\nAI: {error_message}")
        state["messages"].append(AIMessage(content=error_message))
        return state

    state["messages"].append(AIMessage(content=response.text)) 
    print(f"\nAI: {response.text}")
    print("CURRENT STATE: ", state["messages"])

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()


conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")


with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")
