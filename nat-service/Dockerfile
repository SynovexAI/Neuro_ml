# NAT requires Python 3.11-3.13 (not 3.14). Pin 3.12.
FROM python:3.12-slim

WORKDIR /app

# System deps some NAT extras need to build wheels.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
