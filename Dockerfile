FROM python:3.12-slim

LABEL org.opencontainers.image.title="dqfs" \
      org.opencontainers.image.description="Data Quality Factor Scoring - deep-ML quality gates with GitHub-native reporting" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir .

# Demo database can be generated at runtime: `dqfs demo`
ENTRYPOINT ["dqfs"]
CMD ["--help"]
