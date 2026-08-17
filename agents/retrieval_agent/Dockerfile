FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py store.py .

# The index lives here. In Kubernetes a PersistentVolume is mounted at this path,
# which is what lets the corpus survive a pod restart - unlike the stateless
# agents, this container's filesystem is not disposable.
ENV RETRIEVAL_DB_PATH=/data/retrieval.db
VOLUME /data

# MCP protocol port + Prometheus metrics port
EXPOSE 8000 9101

CMD ["python", "server.py"]
