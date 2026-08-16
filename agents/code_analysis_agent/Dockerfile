FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# MCP protocol port + Prometheus metrics port
EXPOSE 8000 9102

CMD ["python", "server.py"]
