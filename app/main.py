import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("shipping_delay_api")

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model.joblib")
METADATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "metadata.json")

# Global variables for model state
model_artifact = None
model_metadata = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_artifact, model_metadata
    logger.info("Initializing FastAPI application and loading ML model artifact...")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        logger.error(f"Model artifact or metadata missing at {MODEL_PATH}")
        raise FileNotFoundError("Model artifact missing! Please run src/train.py before starting server.")
        
    try:
        model_artifact = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, 'r') as f:
            model_metadata = json.load(f)
        logger.info(f"Successfully loaded model: {model_metadata.get('best_model')} with threshold {model_metadata.get('optimal_threshold')}")
    except Exception as e:
        logger.error(f"Error loading model artifact: {e}")
        raise e
        
    yield
    
    logger.info("Shutting down FastAPI application...")
    model_artifact = None
    model_metadata = None

app = FastAPI(
    title="API Prediksi Keterlambatan Pengiriman Paket (Kasus A)",
    description="REST API End-to-End Machine Learning untuk memprediksi risiko keterlambatan pengiriman paket e-commerce/logistik.",
    version="1.0.0",
    lifespan=lifespan
)

class ShippingPredictionRequest(BaseModel):
    Warehouse_block: Literal['A', 'B', 'C', 'D', 'F'] = Field(
        ..., description="Blok gudang keberangkatan (A, B, C, D, F)"
    )
    Mode_of_Shipment: Literal['Ship', 'Flight', 'Road'] = Field(
        ..., description="Moda transportasi pengiriman (Ship, Flight, Road)"
    )
    Customer_care_calls: int = Field(
        ..., ge=1, le=15, description="Jumlah panggilan customer care (1-15)"
    )
    Customer_rating: int = Field(
        ..., ge=1, le=5, description="Rating pelanggan (1-5)"
    )
    Cost_of_the_Product: float = Field(
        ..., gt=0, le=5000, description="Harga produk dalam USD ($)"
    )
    Prior_purchases: int = Field(
        ..., ge=1, le=30, description="Banyak pembelian sebelumnya oleh pelanggan"
    )
    Product_importance: Literal['low', 'medium', 'high'] = Field(
        ..., description="Prioritas/pentingnya produk (low, medium, high)"
    )
    Gender: Literal['F', 'M'] = Field(
        ..., description="Jenis kelamin pelanggan (F, M)"
    )
    Discount_offered: float = Field(
        ..., ge=0, le=100, description="Diskon yang diberikan dalam USD ($)"
    )
    Weight_in_gms: float = Field(
        ..., ge=100, le=15000, description="Berat total paket dalam gram (100-15000 gms)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Warehouse_block": "F",
                "Mode_of_Shipment": "Flight",
                "Customer_care_calls": 4,
                "Customer_rating": 3,
                "Cost_of_the_Product": 210.0,
                "Prior_purchases": 3,
                "Product_importance": "high",
                "Gender": "F",
                "Discount_offered": 5.0,
                "Weight_in_gms": 4500.0
            }
        }
    }

class ShippingPredictionResponse(BaseModel):
    status: str
    prediction_label: str
    is_delayed: bool
    delay_probability: float
    risk_level: str
    applied_threshold: float
    recommendation: str
    model_version: str

@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "Paket Logistics Delay Prediction API",
        "kasus": "Kasus A — Klasifikasi Prediksi Keterlambatan Pengiriman Paket",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    is_loaded = model_artifact is not None and model_metadata is not None
    if not is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded."
        )
    return {
        "status": "ok",
        "model_loaded": True,
        "algorithm": model_metadata.get("best_model", "unknown"),
        "optimal_threshold": model_metadata.get("optimal_threshold", 0.50)
    }

@app.post(
    "/predict-keterlambatan",
    response_model=ShippingPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"]
)
async def predict_delay(payload: ShippingPredictionRequest):
    if model_artifact is None or model_metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model server is initializing or model is missing."
        )
        
    try:
        # Convert request payload to Pandas DataFrame matching input structure
        input_data = pd.DataFrame([{
            'Warehouse_block': payload.Warehouse_block,
            'Mode_of_Shipment': payload.Mode_of_Shipment,
            'Customer_care_calls': payload.Customer_care_calls,
            'Customer_rating': payload.Customer_rating,
            'Cost_of_the_Product': payload.Cost_of_the_Product,
            'Prior_purchases': payload.Prior_purchases,
            'Product_importance': payload.Product_importance,
            'Gender': payload.Gender,
            'Discount_offered': payload.Discount_offered,
            'Weight_in_gms': payload.Weight_in_gms
        }])
        
        # Calculate raw probability from ML Pipeline
        prob_delayed = float(model_artifact.predict_proba(input_data)[0, 1])
        
        # Apply cost-optimal threshold
        optimal_threshold = float(model_metadata.get("optimal_threshold", 0.25))
        is_delayed = bool(prob_delayed >= optimal_threshold)
        
        # Determine risk level & recommendation
        if prob_delayed >= 0.70:
            risk_level = "SANGAT TINGGI"
            recommendation = "Prioritaskan segera ke jalur ekspres premium untuk menghindari klaim keterlambatan."
        elif prob_delayed >= optimal_threshold:
            risk_level = "MODERAT / RISIKO TERLAMBAT"
            recommendation = "Tandai paket untuk pemantauan rute dan optimasi penanganan gudang."
        else:
            risk_level = "RENDAH (AMANCAN)"
            recommendation = "Proses melalui pengiriman standar reguler."
            
        prediction_label = "terlambat" if is_delayed else "tepat_waktu"
        
        logger.info(
            f"Prediction processed: prob={prob_delayed:.4f}, label={prediction_label}, mode={payload.Mode_of_Shipment}"
        )
        
        return ShippingPredictionResponse(
            status="success",
            prediction_label=prediction_label,
            is_delayed=is_delayed,
            delay_probability=round(prob_delayed, 4),
            risk_level=risk_level,
            applied_threshold=optimal_threshold,
            recommendation=recommendation,
            model_version=f"{model_metadata.get('best_model')}-v1.0"
        )
        
    except Exception as e:
        logger.error(f"Error during prediction inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
