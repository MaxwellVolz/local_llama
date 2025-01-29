
# local_llama

LLM Chat API with FastAPI & Llama.cpp (Mistral 7B)

This project runs a **local AI assistant** on a CPU-only machine using:
- `llama.cpp` for **optimized CPU inference**
- `Mistral 7B (q4_0)` GGUF model
- `FastAPI` for API hosting


- [local\_llama](#local_llama)
  - [Installation \& Setup](#installation--setup)
    - [1. Download the Model](#1-download-the-model)
    - [2. Build and Run in Docker](#2-build-and-run-in-docker)
    - [3. Test the Chat API](#3-test-the-chat-api)


## Installation & Setup

### 1. Download the Model
Before building the Docker image, **download the model**:- [local\_llama](#local_llama)


```bash
mkdir models
cd models
wget https://huggingface.co/TheBloke/Mistral-7B-GGUF/resolve/main/mistral-7b-q4_0.gguf
```

### 2. Build and Run in Docker

```bash
docker-compose up --build -d
```
### 3. Test the Chat API

```bash
curl "http://localhost:8000/chat?prompt=Hello"
```