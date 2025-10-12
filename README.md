# rag-evals-starter

[![CI](https://github.com/your-org/rag-evals-starter/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/rag-evals-starter/actions/workflows/ci.yml)

A reproducible Retrieval‑Augmented Generation (RAG) evaluation pipeline that demonstrates how to measure retrieval quality and answer faithfulness useing GCP pgvector.  This repository contains a small demo rag application, an example dataset, a simple RAG pipeline and unit tests.  The goal is to make it straightforward to reproduce experiments and iterate on improvements.

## 🚀 Prerequisites

Before running the ingestion or evaluation scripts, make sure you have the following configured.

### 🧩 System Requirements
* Python 3.11+

* PostgreSQL 14+ with the pgvector extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
* .evn file with 
> 💡 Copy `.env.example` → `.env` and update it with your own credentials.

## 🛠️ Installation

Follow these steps to set up the project locally.

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Danuka97/rag-evals-starter.git
cd rag-evals-starter
```
### 3️⃣ Install Dependencies (using [uv](https://github.com/astral-sh/uv))

Sync and install all dependencies:

```bash
uv sync
```

```bash
source .venv/bin/activate
```

```bash
uv add package-name
```

## Overview

This starter project provides a minimal yet fully structured example of a RAG evaluation pipeline.  It includes:

* **Standard project structure** following the cookie‑cutter data science convention (see `src/`, `tests/`, `data/`).
* **Example dataset** (`data/sample_input.json`) for quickly running the pipeline.
* **Pipeline implementation** (`src/pipeline.py`) that reads a list of questions, performs dummy retrieval and evaluation, and writes metrics to CSV.
* **Unit tests** (`tests/test_pipeline.py`) ensuring the pipeline runs end‑to‑end.
* **Demo application** (`demo/demo_app.py`) that invokes the pipeline and prints where the metrics are saved.
* **Dockerfile** for containerised execution.
* **GitHub Actions CI** to run tests on every push.
* **Pre‑commit configuration** for code quality checks.

## Quickstart

To clone and run the project locally:

```bash
git clone https://github.com/your-org/rag-evals-starter.git
cd rag-evals-starter

# Set up a virtual environment and install dependencies
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Run unit tests
pytest -q

# Run the pipeline on the sample data
python src/pipeline.py --input data/sample_input.json --output output/metrics.csv

# Inspect the resulting metrics
cat output/metrics.csv
```

Alternatively you can run everything in Docker:

```bash
docker build -t rag-evals-starter .
docker run --rm -v "$PWD/output":/app/output rag-evals-starter
```

## Architecture

The high‑level flow of the pipeline is as follows:

1. Load questions from a JSON file (`data/sample_input.json`).
2. For each question, perform a simple retrieval step (dummy in this starter project).  In a full implementation you would query a vector database, run a language model to generate answers, and possibly apply re‑ranking.
3. Evaluate the quality of the retrieved context and the generated answer.  Here we stub out metrics such as *precision* and *faithfulness*.
4. Write the metrics to a CSV file (`output/metrics.csv`).

You can extend `src/pipeline.py` to integrate real retrieval tools (e.g. FAISS, LlamaIndex) and evaluation frameworks (e.g. Ragas) and add ablations such as HyDE or rerankers.  The current code is deliberately simple to serve as a template.

## Benchmarks (placeholder)

In a real RAG system you would experiment with different retrieval strategies and measure their impact on precision, faithfulness, latency and cost.  Below is a placeholder table; fill it with your own results as you iterate.

| Configuration            | Precision | Faithfulness | Notes                  |
|--------------------------|----------:|-------------:|------------------------|
| Baseline                 |     0.60  |         0.75 | Dummy metrics only     |
| Baseline + HyDE          |     0.68  |         0.79 | Example improvement    |
| Baseline + Reranker      |     0.65  |         0.82 | Example improvement    |
| HyDE + Reranker Combined |     0.72  |         0.85 | Example improvement    |

## Roadmap

This project is a starting point.  Possible next steps:

* Integrate a vector database (FAISS, Chroma, Pinecone or managed service).
* Use LangChain, LlamaIndex or LangGraph to build the retrieval pipeline.
* Compute real evaluation metrics with [Ragas](https://github.com/explodinggradients/ragas) or similar libraries.
* Create a Streamlit or FastAPI demo application to interactively explore results.
* Add more comprehensive unit tests and performance monitoring.

## License

This project is released under the MIT License.  Feel free to use it as a template for your own RAG experiments.
