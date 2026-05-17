"""Interactive LangGraph chatbot backed by Google GenAI."""

import os
import sys
import time
from typing import List, TypedDict

from dotenv import (
    load_dotenv,  # used to store secret stuff like API keys or configuration values
)
from google import genai
from google.genai import errors
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langsmith import wrappers

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
class AgentState(TypedDict):
    """State passed through the LangGraph workflow."""

    messages: List[HumanMessage]

MODEL = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))


def generate_with_retry(prompt: str):
    """Generate content with app-level retries for transient Gemma 500s."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=prompt,
            )
        except errors.ServerError:
            if attempt == MAX_RETRIES:
                raise
            print(f"\nGemma returned a server error. Retrying ({attempt}/{MAX_RETRIES})...")  # noqa: T201
            time.sleep(attempt)

def process(state: AgentState) -> AgentState:
    """Send the latest user message to the configured GenAI model."""
    user_message = state["messages"][-1].content
    try:
        response = generate_with_retry(user_message)
    except errors.ServerError as exc:
        print(f"\nGoogle GenAI server error while using {MODEL}: {exc}")  # noqa: T201
        print("The model is reachable, but Google's Gemma endpoint returned 500 after retries.")  # noqa: T201
        return state
    except errors.APIError as exc:
        print(f"\nGoogle GenAI API error while using {MODEL}: {exc}")  # noqa: T201
        return state

    print(f"\nAI: {response.text}")  # noqa: T201
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()

user_input = input("Enter: ")
while user_input.lower() != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
