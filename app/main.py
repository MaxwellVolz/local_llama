from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import os
import time
import logging
from datetime import datetime

# Load environment variables
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

if not huggingface_token:
    raise ValueError("Hugging Face token not found! Make sure it's set in the environment.")

# Authenticate Hugging Face
login(token=huggingface_token)

# Initialize FastAPI
app = FastAPI()

# Model Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Change this for different models
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)

# Create Log Folder Structure
LOG_DIR = f"logs/{MODEL_NAME.replace('/', '_')}"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# Configure Logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

logging.info(f"Model loaded: {MODEL_NAME}")


@app.get("/chat")
def chat(prompt: str):
    """
    Handles user input and generates a response using the Meta Llama model.
    Logs request duration, input, and response.
    """
    start_time = time.time()  # Track request start time

    try:
        logging.info(f"Received prompt: {prompt}")

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=50)

        # Decode response
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        request_duration = round(time.time() - start_time, 3)

        logging.info(f"Response: {response_text}")
        logging.info(f"Request Duration: {request_duration} sec")

        return {
            "response": response_text,
            "duration": request_duration,
            "model": MODEL_NAME
        }

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}


