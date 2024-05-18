from fastapi import FastAPI, UploadFile, File, Request
import uvicorn
from tensorflow import keras
import numpy as np
import os
from PIL import Image
import io
import psutil
import time
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge

API_COUNTER = Counter("ApiUsageCount", "Api Invocation Counter", ["client_ip"])
CPU_TIME= Gauge("CpuUsageTime", "Cpu Utilization Duration", ["client_ip"])
MEMORY_UTIL_GAUGE = Gauge("MemoryUtilGauge", "Memory utilization", ["client_ip"])

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Function to load the trained model
def load_model(path: str) -> keras.Sequential:
    model = keras.models.load_model(path)
    return model

# Load the model from the path provided as environment arguement
model_path = os.getenv("MODEL_PATH")
final_model = load_model(model_path)

# Function to format the image to match the input size of the model
def format_image(img):
    img_array = np.array(img.resize((28, 28))) # Resize the image to 28x28
    return img_array

# Function to predict the digit from the formatted image data
def predict_digit(model, data_point: list) -> str:
    data = np.array(data_point).reshape(-1, 784) / 255.0 # Normalize the data
    prediction = model.predict(data) # Make predictions
    digit = np.argmax(prediction) # Get the index of the highest prediction
    return str(digit)

# Root(bootup) endpoint
@app.get("/")
def read_root():
    return {"This is an": "MNIST app"}

# Function to calculate processing time
def unit_processing_time(start_time: float, length: int) -> float:
    # Calculate processing time per character
    end_time = time.time()
    total_time = end_time - start_time
    upt = (total_time/length)*1e6 
    return upt

# Prediction endpoint
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    start_time = time.time() # saving the start time
    contents = await file.read()  # Read the uploaded file
    client_ip = request.client.host
    API_COUNTER.labels(client_ip=client_ip).inc() #Count the API calls
    img = Image.open(io.BytesIO(contents)).convert('L')  # Open the image and convert to grayscale
    img_array = format_image(img)  # Format the image
    data_point = img_array.flatten().tolist()  # Flatten the image array and convert to list
    cpu_usage=psutil.cpu_percent(interval=1) #The CPU usage
    CPU_TIME.labels(client_ip=client_ip).set(cpu_usage)
    memory_util = psutil.virtual_memory() #The memory information
    MEMORY_UTIL_GAUGE.labels(client_ip=client_ip).set(memory_util.percent) #Set the memory util
    digit = predict_digit(final_model, data_point)  # Predict the digit
    return {"digit": digit}  # Return the predicted digit as response

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
