"""
Pipeline utilities to run question answering against a PGVector store.

This module reads a dataframe of questions, retrieves context from PGVector
using Vertex AI embeddings, generates answers with a Vertex AI chat model, and
returns an enriched dataframe. A small CLI is provided for convenience.
"""

import os
import pandas as pd
import argparse
from dotenv import load_dotenv
from urllib.parse import quote_plus
from rag_evals_starter.core import df_to_documents, collection_exists, verify_ingest

from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_google_vertexai import VertexAIEmbeddings

from langchain_google_vertexai import ChatVertexAI
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Rag script")

    # Define arguments
    parser.add_argument("--csv_file_path", type=str, required=True, help="path to the CSV file with questions")
    parser.add_argument("--column_name", type=str, help="column with questions")
    parser.add_argument("--file_name",type=str, help="file name")
    parser.add_argument("--save_location",type=str, help="save location of file")


    # Parse the arguments
    args = parser.parse_args()
    load_dotenv()

    # Use the arguments

    # Load the dataset)
    emb = VertexAIEmbeddings(model_name="text-embedding-005", project=os.getenv("GCP_PROJECT"), location=os.getenv("GCP_LOCATION"))
    pw = quote_plus(os.getenv("SQL_password"))
    CONNECTION_STRING = f"postgresql+psycopg2://{os.getenv('SQL_user')}:{pw}@127.0.0.1:5432/{os.getenv('SQL_datbase')}"


    vectorstore = PGVector.from_existing_index(
    embedding=emb,
    collection_name="my_chunks",
    connection_string=CONNECTION_STRING,)

    def join_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def build_q_and_context(q: str):
        docs = retriever.get_relevant_documents(q)
        context = join_docs(docs)
        return {"question": q, "context": context}

    df = pd.read_csv(args.csv_file_path)

    if args.column_name not in df.columns:
        raise ValueError(f"Column '{args.column_name}' not found in CSV. Available: {list(df.columns)}")


    queations_list = df.question.tolist()


    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # batched_results = retriever.batch(queations_list)

    llm = ChatVertexAI(
    model_name="gemini-2.5-flash",  # or the Vertex model you have access to
    temperature=0.0,
    )

    build_qc = RunnableLambda(build_q_and_context)


    prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use the context to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "If you don't know, say you don’t know."
    )

    rag_chain = (
    build_qc
    | prompt
    | llm
    | StrOutputParser()
    )

    print("---------------------------------------------------Batch processing for context started----------------------------------")
    contexts_and_qs = build_qc.batch(queations_list)
    # contexts_and_qs is a list of dicts: [{"question": q, "context": c}, ...]

    print("---------------------------------------------------Batch processing for context done----------------------------------")

    contexts = [d["context"] for d in contexts_and_qs]

    print("---------------------------------------------------Batch processing for answers started ----------------------------------")

    # 2. Batch get answers
    answers = rag_chain.batch(queations_list)

    print("---------------------------------------------------Batch processing for answers done----------------------------------")

    df["retrive_context"] = contexts
    df["answer"] = answers
    df.to_csv(f"{args.save_location}/{args.file_name}.csv", index=False)
    print(f"save file with retiver context to {args.save_location}/{args.file_name}.csv")


if __name__ == "__main__":
    main()