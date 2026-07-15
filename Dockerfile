FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
COPY configs ./configs
COPY data ./data
RUN python -m pip install --upgrade pip setuptools && python -m pip install .
COPY docs ./docs
COPY scripts ./scripts

ENTRYPOINT ["bc250"]
CMD ["--help"]
