"""Insurance RAG CLI agent for Great Eastern report analysis.

The agent can answer questions from the PDFs in ``rag_input/``, perform basic
numeric calculations, draft/update report text, save reports as ``.txt`` files,
and keep the current terminal session as conversation memory.
"""

# ruff: noqa: T201

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Annotated, Sequence

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

PDF_FOLDER = Path("rag_input")
PERSIST_DIRECTORY = Path("chroma_db")
COLLECTION_NAME = "insurance_reports"
REPORTS_DIRECTORY = Path("reports")
LOG_FILE = Path("logging.txt")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
EMBEDDING_MODEL = "models/gemini-embedding-001"

report_content = ""

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


class AgentState(TypedDict):
    """Conversation state passed through the LangGraph workflow."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


class KeywordRetriever:
    """Local fallback retriever used when embedding quota is exhausted."""

    def __init__(self, documents: Sequence[Document], k: int = 5) -> None:
        """Store document chunks and the number of chunks to return."""
        self.documents = list(documents)
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        """Return the most relevant chunks using simple keyword overlap."""
        query_terms = tokenize(query)
        if not query_terms:
            return self.documents[: self.k]

        scored_documents = []
        for doc in self.documents:
            content_terms = tokenize(doc.page_content)
            score = sum(content_terms.count(term) for term in query_terms)
            if score > 0:
                scored_documents.append((score, doc))

        scored_documents.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_documents[: self.k]]


def tokenize(text: str) -> list[str]:
    """Split text into lowercase search terms for local fallback retrieval."""
    return [
        term
        for term in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(term) > 2 and term not in STOP_WORDS
    ]


def load_pdf_documents() -> list[Document]:
    """Load and split all PDF files from the configured RAG input folder."""
    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in the folder: {PDF_FOLDER}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []

    for pdf_path in pdf_files:
        try:
            pdf_loader = PyPDFLoader(str(pdf_path))
            pages = pdf_loader.load()
            print(f"Loaded '{pdf_path.name}' with {len(pages)} pages.")
            documents.extend(text_splitter.split_documents(pages))
        except Exception as exc:
            print(f"Error loading '{pdf_path.name}': {exc}")
            raise

    return documents


def get_collection_count(vectorstore: Chroma) -> int:
    """Return the number of chunks stored in Chroma."""
    try:
        return vectorstore._collection.count()
    except Exception:
        return 0


def create_retriever():
    """Create a Chroma retriever, reusing persisted chunks when available."""
    PERSIST_DIRECTORY.mkdir(exist_ok=True)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIRECTORY),
        )
        stored_chunks = get_collection_count(vectorstore)
        if stored_chunks > 0:
            print(f"Loaded existing Chroma vectorstore with {stored_chunks} chunks.")
            return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    except Exception as exc:
        print(f"Error loading Chroma vectorstore: {exc}")
        vectorstore = None

    documents = load_pdf_documents()

    if vectorstore is not None:
        try:
            vectorstore.add_documents(documents)
            print(f"Chroma vectorstore created with {len(documents)} chunks.")
            return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        except Exception as exc:
            print(f"Error creating Chroma vectorstore: {exc}")

    print("Falling back to local keyword retrieval. Semantic search will be less precise.")
    return KeywordRetriever(documents, k=5)


retriever = create_retriever()


@tool
def add(a: float, b: float) -> float:
    """Add two numbers. Use this for arithmetic requests that require addition."""
    print("Calculating addition...")
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a. Use this for arithmetic requests that require subtraction."""
    print("Calculating subtraction...")
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers. Use this for arithmetic requests that require multiplication."""
    print("Calculating multiplication...")
    return a * b


@tool
def update(content: str) -> str:
    """Replace the current report draft with the complete provided report content."""
    global report_content

    print("Updating report draft...")
    report_content = content.strip()

    if not report_content:
        return "The report draft is empty. Please provide report content before saving."

    return "Report draft updated successfully."


@tool
def save(filename: str) -> str:
    """Save the current report draft as a UTF-8 .txt file in the reports folder."""
    global report_content

    print("Saving report...")
    if not report_content.strip():
        return "No report content is available to save. Use update first."

    safe_name = Path(filename.strip() or "report").name
    if not safe_name.lower().endswith(".txt"):
        safe_name = f"{safe_name}.txt"

    REPORTS_DIRECTORY.mkdir(exist_ok=True)
    output_path = REPORTS_DIRECTORY / safe_name

    try:
        output_path.write_text(report_content, encoding="utf-8")
        return f"Report saved successfully to '{output_path}'."
    except Exception as exc:
        return f"Error saving report: {exc}"


@tool
def retriever_tool(query: str) -> str:
    """Search the insurance report PDFs and return relevant excerpts with source citations."""
    print("Retrieving information from insurance reports...")
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the insurance reports."

    results = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = Path(str(metadata.get("source", "unknown source"))).name
        page = metadata.get("page")
        page_label = f"page {int(page) + 1}" if isinstance(page, int) else "page unknown"

        results.append(
            f"Source {index}: {source}, {page_label}\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n".join(results)


tools = [add, subtract, multiply, update, save, retriever_tool]
model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0).bind_tools(tools)

system_prompt = """
You are an insurance expert assistant for Great Eastern report analysis.

Capabilities:
- Use retriever_tool to answer questions about the loaded insurance reports.
- Use add, subtract, and multiply for arithmetic instead of calculating silently.
- Use update with the full report text when creating or changing a report draft.
- Use save when the user wants to save the current report as a .txt file.

Rules:
- For questions about the insurance reports, retrieve evidence before answering.
- Cite the source filename and page from retriever_tool results when available.
- If the reports do not contain enough evidence, say what is missing.
- Keep answers clear and concise unless the user asks for a longer report.
- Treat the conversation history in this session as memory.
"""


def should_continue(state: AgentState) -> str:
    """Route to tools when the latest model message contains tool calls."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"


def call_llm(state: AgentState) -> AgentState:
    """Call the model with the system prompt and full session message history."""
    print("LLM is running...")
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = model.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "llm")

agent = graph.compile()


def message_content_to_text(content: object) -> str:
    """Convert LangChain message content into readable log text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        block_type = content.get("type")
        if block_type in {"thinking", "reasoning"}:
            return ""
        if "text" in content:
            return str(content["text"])
        return ""
    if isinstance(content, list):
        text_parts = []
        for item in content:
            text = message_content_to_text(item).strip()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts)
    return str(content)


def write_conversation_log(messages: Sequence[BaseMessage], log_file: Path = LOG_FILE) -> None:
    """Write the current session transcript to a readable text log."""
    lines = ["Insurance RAG Agent Conversation Log", ""]

    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message_content_to_text(message.content)}")
        elif isinstance(message, AIMessage):
            content = message_content_to_text(message.content).strip()
            if content:
                lines.append(f"AI: {content}")
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    lines.append(f"AI tool call: {tool_call['name']}({tool_call.get('args', {})})")
        elif isinstance(message, ToolMessage):
            lines.append(f"Tool {message.name}: {message_content_to_text(message.content)}")
        else:
            lines.append(f"{message.type}: {message_content_to_text(message.content)}")

        lines.append("")

    lines.append("End of Conversation")
    log_file.write_text("\n".join(lines), encoding="utf-8")


def latest_ai_response(messages: Sequence[BaseMessage]) -> str:
    """Return the latest non-empty assistant message content."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message_content_to_text(message.content).strip()
            if content:
                return content
    return ""


def run_agent() -> None:
    """Run the insurance RAG agent as a terminal chatbot."""
    print("\n=== INSURANCE RAG AGENT ===")
    print("Type 'exit' or 'quit' to stop and save the session log.")

    conversation_history: list[BaseMessage] = []

    try:
        while True:
            user_input = input("\nWhat is your question: ").strip()

            if user_input.lower() in {"exit", "quit"}:
                break

            if not user_input:
                continue

            conversation_history.append(HumanMessage(content=user_input))
            result = agent.invoke({"messages": conversation_history})
            conversation_history = list(result["messages"])

            response = latest_ai_response(conversation_history)
            if response:
                print("\n=== ANSWER ===")
                print(response)
    finally:
        write_conversation_log(conversation_history)
        print(f"\nConversation log saved to {LOG_FILE}")


if __name__ == "__main__":
    run_agent()
