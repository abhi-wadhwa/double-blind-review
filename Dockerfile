FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
COPY examples/ examples/
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Run tests to verify build
RUN pytest tests/ -v --tb=short

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: run Streamlit app
CMD ["streamlit", "run", "src/viz/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
