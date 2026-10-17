FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
# CPU-only torch for the cross-encoder reranker (the default index pulls CUDA builds)
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py chunk.py ./
COPY regs ./regs
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
