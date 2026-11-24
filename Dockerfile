# CPU image by default. For GPU, switch the base to a CUDA-enabled PyTorch image
# and run with: docker run --gpus all -e CTBODY_DEVICE=gpu ...
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir .

ENV CTBODY_DATA=/data CTBODY_DEVICE=cpu
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "ctbody.api:app", "--host", "0.0.0.0", "--port", "8000"]
