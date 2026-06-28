from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.staticfiles import StaticFiles
app = FastAPI()

# CORS (important for frontend JS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load ML model

model = pickle.load(open("model.pkl", "rb"))
# Input schema
class HeartDiseaseInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float
# UI route (HOME PAGE)
@app.get("/", response_class=HTMLResponse)
def ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


# Prediction API
@app.post("/predict")
def predict(data: HeartDiseaseInput):

    input_data = np.array([[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]])

    prediction = model.predict(input_data)[0]

    result = "❤️ Heart Disease Detected" if prediction == 1 else "💚 No Heart Disease"

    return {"prediction": result}
