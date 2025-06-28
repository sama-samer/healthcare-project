from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

# Load models
diabetes_model_bundle = joblib.load("diabetes_model_with_scaler.joblib")
diabetes_model = diabetes_model_bundle['model']
diabetes_scaler = diabetes_model_bundle['scaler']

# If your file is heart_model.joblib
heart_model_bundle = joblib.load("heart_model.joblib")
heart_model = heart_model_bundle  # fixed!

# Define input schemas
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DPF: float
    Age: int
    
class HeartInput(BaseModel):
    cp: int
    thal: int
    ca: int
    chol: float
    oldpeak: float
    age: int
    thalach: int

# FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the Health Prediction API"}


@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    # Step 1: Prepare features
    features = [data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness,
                data.Insulin, data.BMI, data.DPF, data.Age]
    input_array = np.array([features])
    input_scaled = diabetes_scaler.transform(input_array)

    # Step 2: Make prediction
    prediction = diabetes_model.predict(input_scaled)[0]
    result = "Diabetic" if prediction else "Non-Diabetic"

    # Step 3: Anomaly detection on Glucose + BP
    detection_features = np.array([[data.Glucose, data.BloodPressure]])

    # Simulated IsolationForest (you can load a trained model instead)
    normal_data = np.array([
        [90, 70], [100, 80], [110, 75], [95, 72],
        [105, 78], [85, 65], [120, 85], [100, 70]
    ])
    iso_model = IsolationForest(contamination=0.2, random_state=42)
    iso_model.fit(normal_data)
    is_anomaly = iso_model.predict(detection_features)[0] == -1

    # Step 4: Return JSON-serializable values
    return {
        "prediction": int(prediction),               # Ensure it's int (not np.int64)
        "result": result,                            # Already a string
        "anomaly_detected": bool(is_anomaly),        # Ensure it's a Python bool
        "note": "Anomaly means the glucose or BP value is unusual."
    }
@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    input_array = np.array([[data.cp, data.thal, data.ca, data.chol,
                             data.oldpeak, data.age, data.thalach]])
    prediction = heart_model.predict(input_array)[0]
    return {"prediction": int(prediction), "result": "At Risk" if prediction else "No Risk"}