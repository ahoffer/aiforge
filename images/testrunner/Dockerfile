FROM python:3.13-slim

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
COPY images/testrunner/logfmt /app/logfmt
COPY images/testrunner/pytest-compact /app/pytest-compact
RUN chmod +x /app/run.sh /app/logfmt /app/pytest-compact /app/tests/*.sh

ENTRYPOINT ["/app/run.sh"]
