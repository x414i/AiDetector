FROM python:3.13.2 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Create a virtual environment
RUN python -m venv .venv
COPY requirements.txt ./
# Install requirements in the virtual environment
RUN .venv/bin/pip install -r requirements.txt

FROM python:3.13.2-slim
# Install PostgreSQL client libraries
RUN apt-get update && apt-get install -y libpq5 && apt-get clean

WORKDIR /app
# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv .venv/
COPY . .
# Start the Flask application
CMD ["/app/.venv/bin/flask", "run", "--host=0.0.0.0", "--port=8080"]