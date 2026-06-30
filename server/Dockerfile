FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency install
RUN pip install --no-cache-dir uv

COPY pyproject.toml .
# Install only server dependencies (no torch training stack)
RUN uv pip install --system --no-cache \
    fastapi uvicorn[standard] websockets \
    onnxruntime numpy torch torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY server/ server/
COPY model/onnx/ model/onnx/

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
