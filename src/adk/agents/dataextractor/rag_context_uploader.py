#!/usr/bin/env python3
"""
rag_context_uploader.py — Minimal, clean runner to export context text to TXT and upload to GCS.

Industrial touches:
- Clear CLI with argparse and helpful errors.
- Uses Hugging Face `datasets` to read a dataset split.
- Streams rows (low memory), writes .txt shards under a size cap (default <10MB each).
- Uploads to Google Cloud Storage with the official client and content-type.
- Works with ADC or an explicit service-account key.

Example
-------
python rag_context_uploader.py \
  --hf-dataset neural-bridge/rag-dataset-1200 \
  --split train \
  --context-col context \
  --bucket adk_rag2 \
  --dest-prefix corpora/my-ds \
  --max-bytes 9000000

Notes
-----
- Vertex AI RAG Engine supports TXT and JSONL with a 10 MB/file limit; we default to ~9 MB shards.
- Authenticate with ADC (GOOGLE_APPLICATION_CREDENTIALS) or --key-path.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

# Optional imports done inside functions to keep startup fast


# ------------------------------- config -------------------------------
DEFAULT_SEP = "\n\n---\n\n"
DEFAULT_MAX_BYTES = 9_000_000  # keep under 10 MB limit per text file


@dataclass
class Args:
    hf_dataset: str
    split: str
    context_col: str
    bucket: str
    dest_prefix: str
    project: str | None
    key_path: str | None
    max_bytes: int
    no_headers: bool
    verbose: bool


# ------------------------------ helpers ------------------------------

def _iter_contexts_from_hf(dataset_name: str, split: str, context_col: str) -> Iterator[str]:
    """Yield context strings from a HF dataset split without loading all to RAM."""
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Hugging Face `datasets` is required. Install with: pip install datasets"
        ) from e

    ds = load_dataset(dataset_name, split=split, streaming=True)  # streaming = low mem
    for row in ds:
        val = row.get(context_col, "")
        if val is None:
            continue
        yield str(val)


def _write_txt_shards(contexts: Iterable[str], out_dir: Path, *, max_bytes: int, sep: str, headers: bool) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx, size = 0, 0
    current = out_dir / f"corpus_{shard_idx:03d}.txt"
    fh = current.open("w", encoding="utf-8")
    paths: list[Path] = [current]

    def _roll():
        nonlocal shard_idx, size, fh, current
        fh.close()
        shard_idx += 1
        size = 0
        current = out_dir / f"corpus_{shard_idx:03d}.txt"
        paths.append(current)
        return current.open("w", encoding="utf-8")

    for i, text in enumerate(contexts):
        text = (text or "").strip()
        block = (f"### DOC {i:06d}\n" if headers else "") + text + "\n" + sep
        block_bytes = len(block.encode("utf-8"))
        if size + block_bytes > max_bytes and size > 0:
            fh = _roll()
        fh.write(block)
        size += block_bytes

    fh.close()
    return paths


def _gcs_client(project: str | None, key_path: str | None):
    from google.cloud import storage
    if key_path:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(key_path)
        return storage.Client(project=project, credentials=creds)
    # ADC (env var or gcloud login)
    return storage.Client(project=project)


def _upload_many(paths: list[Path], bucket: str, dest_prefix: str, project: str | None, key_path: str | None) -> list[str]:
    client = _gcs_client(project, key_path)
    b = client.bucket(bucket)
    uris: list[str] = []
    for p in paths:
        rel_name = p.name
        dest = f"{dest_prefix.rstrip('/')}/{rel_name}" if dest_prefix else rel_name
        blob = b.blob(dest)
        blob.content_type = "text/plain"
        blob.upload_from_filename(str(p))
        uris.append(f"gs://{bucket}/{dest}")
        logging.info("Uploaded %s -> gs://%s/%s", p, bucket, dest)
    return uris


# --------------------------------- CLI --------------------------------

def parse_args(argv: list[str]) -> Args:
    ap = argparse.ArgumentParser(description="Export HF contexts to TXT shards and upload to GCS")
    ap.add_argument("--hf-dataset", required=True, help="HF dataset repo (e.g. org/name)")
    ap.add_argument("--split", default="train", help="Dataset split (e.g. train, validation)")
    ap.add_argument("--context-col", default="context", help="Column containing context text")

    ap.add_argument("--bucket", required=True, help="GCS bucket name")
    ap.add_argument("--dest-prefix", default="", help="Destination prefix within the bucket")

    ap.add_argument("--project", default=None, help="GCP project ID (optional)")
    ap.add_argument("--key-path", default=None, help="Service account JSON key path (optional)")

    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="Max bytes per TXT shard")
    ap.add_argument("--no-headers", action="store_true", help="Do not write '### DOC N' headers")
    ap.add_argument("-v", "--verbose", action="store_true")

    ns = ap.parse_args(argv)
    return Args(
        hf_dataset=ns.hf_dataset,
        split=ns.split,
        context_col=ns.context_col,
        bucket=ns.bucket,
        dest_prefix=ns.dest_prefix,
        project=ns.project,
        key_path=ns.key_path,
        max_bytes=ns.max_bytes,
        no_headers=ns.no_headers,
        verbose=ns.verbose,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    # 1) Read contexts
    logging.info("Loading dataset: %s [%s]", args.hf_dataset, args.split)
    contexts = _iter_contexts_from_hf(args.hf_dataset, args.split, args.context_col)

    # 2) Write TXT shards
    out_dir = Path("out_txt")
    paths = _write_txt_shards(contexts, out_dir, max_bytes=args.max_bytes, sep=DEFAULT_SEP, headers=not args.no_headers)

    if not paths:
        logging.error("No files were written; check your dataset/column.")
        return 2

    # 3) Upload to GCS
    uris = _upload_many(paths, args.bucket, args.dest_prefix, args.project, args.key_path)

    print("\nUploaded files:")
    for uri in uris:
        print("  ", uri)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
