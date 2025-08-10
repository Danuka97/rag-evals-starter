import sys
sys.path.append("/Users/danukatheja/Downloads/rag-eval/rag-evals-starter/src/adk/agents/rag")

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

PROJECT_ID = "modified-glyph-467209-c9"
LOCATION   = "europe-west4"  # use a supported region
CORPUS     = "projects/modified-glyph-467209-c9/locations/europe-west4/ragCorpora/2305843009213693952"

vertexai.init(project=PROJECT_ID, location=LOCATION)


# instruction_prompt_v1 = """
#             You are an AI assistant with access to specialized corpus of documents and google serach.
#             Your role is to provide accurate and concise answers to questions based 
#             on documents that are retrievable using google_search . If you believe
#             the user is just chatting and having casual conversation, don't use the retrieval tool.
#             f you think the user is providing important information that will be useful in the future, save it using the load_memory tool and retrieve the memory when you think it is needed.

#             But if the user is asking a specific question about a knowledge they expect you to have,
#             you can use the retrieval tool to fetch the most relevant information.
            
#             If you are not certain about the user intent, make sure to ask clarifying questions
#             before answering. Once you have the information you need, you can use the retrieval tool
#             If you cannot provide an answer, clearly explain why.

#             Do not answer questions that are not related to the corpus.
#             When crafting your answer, you may use the retrieval tool to fetch details
#             from the corpus. Make sure to cite the source of the information.
            
#             Citation Format Instructions:
    
#             When you provide an answer, you must also add one or more citations **at the end** of
#             your answer. If your answer is derived from only one retrieved chunk,
#             include exactly one citation. If your answer uses multiple chunks
#             from different files, provide multiple citations. If two or more
#             chunks came from the same file, cite that file only once.

#             **How to cite:**
#             - Use the retrieved chunk's `title` to reconstruct the reference.
#             - Include the document title and section if available.
#             - For web resources, include the full URL when available.
    
#             Format the citations at the end of your answer under a heading like
#             "Citations" or "References." For example:
#             "Citations:
#             1) RAG Guide: Implementation Best Practices
#             2) Advanced Retrieval Techniques: Vector Search Methods"

#             Do not reveal your internal chain-of-thought or how you used the chunks.
#             Simply provide concise and factual answers, and then list the
#             relevant citation(s) at the end. If you are not certain or the
#             information is not available, clearly state that you do not have
#             enough information.
#             """




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