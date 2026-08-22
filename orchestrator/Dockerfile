FROM python:3.12-slim

WORKDIR /app

# The build context is orchestrator/, matching the agents' pattern - a
# repo-root context would drag .venv and the eval output into the image.
#
# But unlike the agents, which are single self-contained server.py files, this
# is a PACKAGE: api.py imports `orchestrator.config`, `orchestrator.graph` and
# so on. So the context is copied into /app/orchestrator/ rather than /app,
# which puts WORKDIR /app on sys.path and lets those imports resolve. static/
# comes along with it, which is what api.py's FileResponse needs.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/orchestrator/

EXPOSE 8080

# --host 0.0.0.0 is not optional. uvicorn defaults to 127.0.0.1, which inside a
# container means the container's own loopback - the process starts, the port
# looks bound from within, and nothing on the pod network can reach it. The
# readiness probe would fail with the app apparently running fine.
#
# Port 8080 rather than the agents' 8000: nothing forces it, but keeping the
# orchestrator on a different number makes `kubectl get svc` readable at a
# glance. Each pod has its own network namespace, so there is no clash either
# way, and the Windows port-reservation problem does not apply in here.
CMD ["python", "-m", "uvicorn", "orchestrator.api:app", "--host", "0.0.0.0", "--port", "8080"]
