# Downscaling shift test - dashboard and pipeline in one image.
#   docker build -t shifttest .
#   docker run --rm -p 8000:8000 shifttest      -> http://localhost:8000
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    SHIFTTEST_RESULTS=/app/results

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY shifttest ./shifttest
COPY tests ./tests
COPY scripts ./scripts

# Bake the three standard scenarios into the image so the dashboard opens
# with results immediately. Set --build-arg PREBUILD=0 to skip (the server
# then computes them on first start).
ARG PREBUILD=1
RUN if [ "$PREBUILD" = "1" ]; then python -m shifttest sweep --root /app/results; fi

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

CMD ["uvicorn", "shifttest.server:app", "--host", "0.0.0.0", "--port", "8000"]
