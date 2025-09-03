
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load model
model = joblib.load("soil_model.pkl")

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    features: list[float]  # e.g. [5.1, 3.5, 1.4, 0.2]

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    pred = model.predict(X)[0]
    return {"prediction": int(pred)}
