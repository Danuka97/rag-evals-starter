#!/usr/bin/env python
"""
Demo application to run the RAG evaluation pipeline on a sample dataset.
This script runs the pipeline using the default sample input and writes the metrics
to a CSV file for quick inspection.
"""
import argparse
from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Demo app for RAG evaluation pipeline")
    parser.add_argument("--input", default="data/sample_input.json", help="Path to input questions JSON")
    parser.add_argument("--output", default="demo/demo_output.csv", help="Path to output metrics CSV")
    args = parser.parse_args()

    # Run the pipeline with the provided arguments
    run_pipeline(args.input, args.output)

    print(f"Demo complete. Metrics written to {args.output}")


if __name__ == "__main__":
    main()
