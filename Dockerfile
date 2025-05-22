FROM python:3.12-slim

WORKDIR /app/OpenManus

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    wget \
    gnupg \
    openssh-client \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome for browser tools
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN pip install --no-cache-dir uv

# Copy application code
COPY . .

# Install dependencies with uv
RUN uv pip install --system -r requirements.txt

# Install additional API and authentication dependencies
RUN uv pip install --system fastapi uvicorn gunicorn psycopg2-binary PyJWT Jinja2 werkzeug

# Create necessary directories for authentication
RUN mkdir -p logs templates/email

# Set environment variables
ENV PYTHONPATH=/app/OpenManus
ENV PORT=8000

# Expose port
EXPOSE 8000

# Start the API server
CMD gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT} app.api.main:app --workers 4
