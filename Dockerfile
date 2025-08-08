FROM python:3.10-slim as base

# Set up a working directory
WORKDIR /app

# Install dependencies separately to leverage layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Create a directory for output at runtime
RUN mkdir -p output

# Default command runs the pipeline on the sample data
CMD ["python", "src/pipeline.py", "--input", "data/sample_input.json", "--output", "output/metrics.csv"]