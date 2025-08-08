import os
import csv
import unittest
from src.pipeline import run_pipeline

class TestPipeline(unittest.TestCase):
    def test_run_pipeline_creates_output_file(self):
        # Setup
        input_path = 'data/sample_input.json'
        output_path = 'tests/output_metrics.csv'
        # Remove output file if exists
        if os.path.exists(output_path):
            os.remove(output_path)
        # Run pipeline
        run_pipeline(input_path, output_path)
        # Assert output file exists
        self.assertTrue(os.path.exists(output_path))
        # Optionally, read metrics to ensure correct structure
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # ensure there is at least one row
            self.assertGreaterEqual(len(rows), 1)
            # check fieldnames
            self.assertEqual(reader.fieldnames, ['question', 'precision', 'faithfulness'])

if __name__ == '__main__':
    unittest.main()
