import pandas as pd
from langchain.docstore.document import Document

import os
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus

# --- minimal additions ---
import psycopg2
from psycopg2.extras import RealDictCursor

def df_to_documents(df: pd.DataFrame, text_col: str) -> list[Document]:
    """Convert a DataFrame column to a list of LangChain Documents (text only)."""
    return [Document(page_content=text) for text in df[text_col].tolist()]

def build_psycopg2_params():
    # Explicit params for psycopg2 (avoids string replace tricks)
    return dict(
        dbname=os.getenv("SQL_datbase"),
        user=os.getenv("SQL_user"),
        password=os.getenv("SQL_password"),
        host=os.getenv("SQL_host", "127.0.0.1"),
        port=os.getenv("SQL_port", "5432"),
    )


def collection_exists(collection_name: str) -> bool:
    params = build_psycopg2_params()
    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM langchain_pg_collection WHERE name = %s;", (collection_name,))
            return cur.fetchone() is not None


# def verify_ingest(collection_name: str) -> tuple[int, int]:
#     """Return (collection_id, row_count) for the given collection_name."""
#     params = build_psycopg2_params()
#     with psycopg2.connect(**params) as conn:
#         with conn.cursor(cursor_factory=RealDictCursor) as cur:
#             cur.execute("SELECT id FROM langchain_pg_collection WHERE name = %s;", (collection_name,))
#             row = cur.fetchone()
#             if not row:
#                 raise RuntimeError(f"Collection '{collection_name}' not found in langchain_pg_collection.")
#             collection_id = row["id"]

#             cur.execute(
#                 "SELECT COUNT(*) AS n FROM langchain_pg_embedding WHERE collection_id = %s;",
#                 (collection_id,)
#             )
#             count = cur.fetchone()["n"]
#             return collection_id, count
        


def _get_collection_pk_column(conn) -> str:
    """
    Returns 'id' or 'uuid' depending on which column exists
    in langchain_pg_collection for this LangChain/pgvector version.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'langchain_pg_collection'
              AND column_name IN ('id', 'uuid')
            ORDER BY column_name
        """)
        rows = cur.fetchall()
        if not rows:
            raise RuntimeError("Could not find 'id' or 'uuid' in langchain_pg_collection.")
        return rows[0]["column_name"]  # either 'id' or 'uuid'

def verify_ingest(collection_name: str) -> tuple[str, int]:
    """
    Return (collection_pk_value, row_count) for the given collection_name.
    collection_pk_value is the value of the PK (id or uuid).
    """
    params = build_psycopg2_params()
    with psycopg2.connect(**params) as conn:
        pk_col = _get_collection_pk_column(conn)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # fetch PK by name
            cur.execute(
                f"SELECT {pk_col} AS pk FROM langchain_pg_collection WHERE name = %s;",
                (collection_name,)
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError(f"Collection '{collection_name}' not found in langchain_pg_collection.")
            pk_value = row["pk"]

            # embedding table uses 'collection_id' across versions
            cur.execute(
                "SELECT COUNT(*) AS n FROM langchain_pg_embedding WHERE collection_id = %s;",
                (pk_value,)
            )
            count = cur.fetchone()["n"]
            return str(pk_value), count
