FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY images/gateway/requirements.txt /tmp/gateway-requirements.txt
COPY requirements.txt /tmp/root-requirements.txt
RUN pip install --no-cache-dir \
    -r /tmp/gateway-requirements.txt \
    -r /tmp/root-requirements.txt \
    pytest anyio

COPY images/gateway/ /app/images/gateway/
COPY tests/ /app/tests/
COPY pyproject.toml /app/pyproject.toml
COPY images/testrunner/run.sh /app/run.sh
RUN chmod +x /app/run.sh /app/tests/*.sh

ENTRYPOINT ["/app/run.sh"]
