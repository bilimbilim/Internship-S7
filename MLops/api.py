from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

import mlflow.pyfunc
import mlflow

mlflow.set_tracking_uri("http://host.docker.internal:5000")
model = mlflow.pyfunc.load_model("models:/dpe_model@production")

@app.get("/")
def home():
    return {
 "surface_habitable": 100,
 "annee_construction": 1990
}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return {
        "prediction_kwh": float(prediction[0])
    }