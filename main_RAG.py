from dotenv import load_dotenv
import os
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

"""
insurance RAG Agent that can :
1. answer questions based on the content of a GE 2025 annual report(life and general insurance), unit fund annual report, and sustainability report.
2. do basic calculations using tools (addition, subtraction, multiplication).
3. write report relevant to GE and save it as a pdf file.
4. store memory of the conversation and use it to answer future questions.
"""

# Load environment variables from .env file
load_dotenv()

# Initialize the language model
llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0)

# Our Embedding Model - compatible with langchain_google_genai
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
)

#read pdf files from the folder "rag_input"
pdf_folder = "rag_input"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

# Safety measure
if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in the folder: {pdf_folder}")

# Load and split the PDF documents
documents = []
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    try:
        pdf_loader = PyPDFLoader(pdf_path)
        pages = pdf_loader.load()
        print(f"Loaded '{pdf_file}' with {len(pages)} pages.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        pages_split = text_splitter.split_documents(pages)
        documents.extend(pages_split)
    except Exception as e:
        print(f"Error loading '{pdf_file}': {e}")
        raise

# Create Chroma vectorstore
persist_directory = "chroma_db"
collection_name = "insurance_reports"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
try:
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Chroma vectorstore created with collection name: {collection_name}")
except Exception as e:
    print(f"Error creating Chroma vectorstore: {e}")
    raise

# Define tools for calculations
@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b        
@tool
def subtract(a: int, b: int) -> int:
    """Subtracts the second number from the first."""
    return a - b
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

#tools to update and save llm written report
@tool
def update(content: str) -> str:
    """Updates the report with the provided content."""
    return f"Report has been updated successfully! The current content is:\n{content}"  
@tool
def save(filename: str) -> str:
    """Save the current report to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
    try:
        with open(filename, 'w') as file:
            file.write(content)
        print(f"\n💾 Report has been saved to: {filename}")
        return f"Report has been saved successfully to '{filename}'."
    except Exception as e:
        return f"Error saving report: {str(e)}"

# Bind the tools to the language model
tools = [add, subtract, multiply, update, save]
model = ChatGoogleGenerativeAI(model="gemma-4-31b-it").bind_tools(tools)

# Define the agent state and processing function
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# This function will process the user input, retrieve relevant information from the vectorstore, and generate a response using the language model.
def rag_process(state: AgentState) -> AgentState:
    contents = "\n".join(
        f"User: {message.content}" if isinstance(message, HumanMessage)
        else f"AI: {message.content}" for message in state["messages"]
    )

    # Retrieve relevant information from the vectorstore based on the user query
    query = state["messages"][-1].content  # Assuming the last message is the user query
    relevant_docs = vectorstore.similarity_search(query, k=3)  # Retrieve top 3 relevant documents
    relevant_content = "\n".join(doc.page_content for doc in relevant_docs)

    # Generate a response using the language model, incorporating the relevant content
    response = model.invoke([
        SystemMessage(content="You are an insurance expert assistant. Use the following relevant information to answer the user's query."),
        SystemMessage(content=relevant_content),
        HumanMessage(content=query)
    ])

    state["messages"].append(AIMessage(content=response.text))
    print(f"\nAI: {response.text}")
    print("CURRENT STATE: ", state["messages"])

    return state

# Create the state graph for the agent
graph = StateGraph(AgentState)
graph.add_node("rag_process", rag_process)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("rag_process")
graph.add_conditional_edges(
    "rag_process", 
    lambda state: "tool_calls" in state["messages"][-1].__dict__ and len(state["messages"][-1].tool_calls) > 0,
    {
        True: "tools",
        False: END,
    }, 
)

graph.add_edge("rag_process", END)
agent = graph.compile()