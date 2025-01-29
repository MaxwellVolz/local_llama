from fastapi import FastAPI
import subprocess
import os

app = FastAPI()

LLAMA_PATH = "/app/llama.cpp/main"
MODEL_PATH = "/app/models/mistral-7b-q4_0.gguf"

@app.get("/chat")
def chat(prompt: str):
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model file not found"}

    # Run llama.cpp as a subprocess
    result = subprocess.run(
        [LLAMA_PATH, "-m", MODEL_PATH, "-p", prompt, "-t", "8", "--n-predict", "256"],
        capture_output=True,
        text=True
    )

    return {"response": result.stdout.strip()}
