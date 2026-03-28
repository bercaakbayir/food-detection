# Use a stable Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for better caching
COPY pyproject.toml .

# Install Python dependencies using uv
# --system flag ensures dependencies are installed into the system site-packages (since this is a container)
RUN uv pip install --no-cache --system .

# Copy the rest of the application
COPY . .

# Create the result file (optional, but good to have)
RUN touch result.jpg

# Set the default command
ENTRYPOINT ["python", "main.py"]
CMD ["--path", "data/glass.png"]
