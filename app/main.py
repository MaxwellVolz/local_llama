from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()


huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

if not huggingface_token:
    raise ValueError("Hugging Face token not found! Make sure it's set in the environment.")

login(token=huggingface_token)

app = FastAPI()

## Requires Meta official LlaMA 2 access
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Change this for different sizes
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Open Variant
# model_name = "togethercomputer/llama-2-7b-chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)


@app.get("/chat")
def chat(prompt: str):
    """
    Handles user input and generates a response using the Meta Llama model.
    """
    try:
        print(f"Received prompt: {prompt}")

        # Generate a response
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=50)

        # Decode and print
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Response text: {response_text}")

        return {"response": response_text}

    except Exception as e:
        print(f"Error processing request: {e}")
        return {"error": str(e)}
