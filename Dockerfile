FROM ubuntu:22.04

# Install dependencies
RUN apt update && apt install -y git build-essential python3 python3-pip wget curl cmake

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything, assuming `main.py` is inside an `app/` folder
COPY . /app  

# Expose API port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
