FROM ubuntu:22.04

# Install dependencies
RUN apt update && apt install -y git build-essential python3 python3-pip

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone llama.cpp and build it
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && make -j$(nproc)

# Copy the model file (Assumes it's manually downloaded)
COPY models/mistral-7b-q4_0.gguf /app/models/mistral-7b-q4_0.gguf

# Copy FastAPI app
COPY app /app

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
