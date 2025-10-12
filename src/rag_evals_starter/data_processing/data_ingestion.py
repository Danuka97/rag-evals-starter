import os
import pandas as pd
import argparse
from dotenv import load_dotenv
from urllib.parse import quote_plus
from rag_evals_starter.core import df_to_documents, collection_exists, verify_ingest

from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_google_vertexai import VertexAIEmbeddings


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="HF data processing script")

    # Define arguments
    parser.add_argument("--csv_file_path", type=str, required=True, help="path to the CSV file with chunks")
    parser.add_argument("--column_name", type=str, help="column with context")
    # parser.add_argument("--file_name",type=str, help="file name")

    # Parse the arguments
    args = parser.parse_args()
    load_dotenv()

    # Use the arguments

    # Load the dataset)
    emb = VertexAIEmbeddings(model_name="text-embedding-005", project=os.getenv("GCP_PROJECT"), location=os.getenv("GCP_LOCATION"))
    pw = quote_plus(os.getenv("SQL_password"))
    CONNECTION_STRING = f"postgresql+psycopg2://{os.getenv('SQL_user')}:{pw}@127.0.0.1:5432/{os.getenv('SQL_datbase')}"

    df = pd.read_csv(args.csv_file_path)

    docs = df_to_documents(df,args.column_name)


#CONNECTION_STRING = "postgresql+psycopg2://USER:Danuka1997@35.246.59.7: /test"
    COLLECTION_NAME = "my_chunks"  # table/collection name

    # Create (or load) the vector store
    vectorstore = PGVector.from_documents(
        documents=docs,
        embedding=emb,
        create_extension=False,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        distance_strategy="cosine",  # or L2/IP
    )

    existed_before = collection_exists(COLLECTION_NAME)

# --- minimal addition: print created/updated status + verification ---
    status = "updated" if existed_before else "created"
    print(f"✅ Collection '{COLLECTION_NAME}' {status}.")

    try:
        collection_id, count = verify_ingest(COLLECTION_NAME)
        print(f"🔎 Verification: collection_id={collection_id}, rows={count}")
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        print("ℹ️ Ensure the 'vector' extension exists and env vars are correct (user/db/host/port).")


if __name__ == "__main__":
    main()

# Example: neural-bridge/rag-dataset-1200


