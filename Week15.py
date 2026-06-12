import logging
import os
import time
from datetime import datetime
from typing import Dict, List

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Predictive Maintenance API",
    description="ML API for equipment failure prediction",
    version="1.0.0",
)


# Define Request Schema (Fixed Pydantic V2 example warnings using examples=[...])
class MachineData(BaseModel):
    temperature: float = Field(
        ...,
        description="Operating temperature in Celsius",
        examples=[75.5]
    )
    vibration_level: float = Field(
        ...,
        description="Vibration amplitude in mm/s",
        examples=[2.3]
    )
    operating_hours: int = Field(
        ...,
        description="Total running hours of the machine",
        examples=[1200]
    )
    rotational_speed: float = Field(
        ...,
        description="RPM of the equipment",
        examples=[1500.0]
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for Normal, 1 for Failure Risk")
    probability: float = Field(..., description="Confidence score of the prediction")
    timestamp: str = Field(..., description="ISO timestamp of the inference request")


# Load production model
MODEL_NAME = "PredictiveMaintenance"
MODEL_STAGE = "Production"

try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info(f"Attempting to load model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model from MLflow: {e}")
    logger.warning("API starting without a loaded model. Inference endpoints will return 503.")
    model = None


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Verifies the API is running and the ML model is loaded."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: MachineData):
    """Accepts telemetry data and predicts equipment failure risk."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Check server logs for loading errors."
        )

    try:
        start_time = time.time()

        # Convert incoming Pydantic data to a Pandas DataFrame
        input_data = pd.DataFrame([payload.model_dump()])

        # Generate prediction
        prediction_result = model.predict(input_data)

        # Assuming classification model outputting an array or list
        pred = int(prediction_result[0])

        # Mock probability fallback
        prob = 0.95 if pred == 1 else 0.05

        latency = time.time() - start_time
        logger.info(f"Prediction successful. Latency: {latency:.4f}s")

        return PredictionResponse(
            prediction=pred,
            probability=prob,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing model prediction: {str(e)}"
        )