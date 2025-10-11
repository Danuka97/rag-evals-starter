import sys

from google.adk.tools import VertexAiSearchTool
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import google_search
from google.adk.tools import load_memory # Tool to query memory
from prompt.rag_agent import instruction_prompt_v1
import vertexai
from vertexai import rag
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
# from google.adk.code_executors import BuiltInCodeExecutor



vertexai.init(project=PROJECT_ID, location=LOCATION)


def rag_search(query: str, top_k: int = 5):
    """Simple RAG search function."""
    resp = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=CORPUS)],
        text=query,
        # similarity_top_k=top_k,
    )
    
    # Convert to simple list
    results = []
    for context in resp.contexts.contexts:
        results.append({
            "text": context.text,
            "uri": context.source_uri if hasattr(context, 'source_uri') else None
        })
    
    return results
rag_tool = FunctionTool(func=rag_search)



root_agent = LlmAgent(
    model="gemini-2.5-flash", 
    name="rag_agent",
    description=instruction_prompt_v1,
    # instruction and tools will be added next
    tools=[rag_tool]
)
