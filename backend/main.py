"""
FastAPI RESTful Service for Crop Disease Advisory System
Author: Nguyen Quoc Khanh (Backend Developer & Database Architect)
SIC Capstone Project - Group 5
"""

import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.config import CONFIDENCE_THRESHOLD, CLASS_NAMES
from backend.inference_engine import (
    CropDiseaseInferenceEngine,
    validate_image_file,
    InputValidationError,
)
from backend.advisory_engine import AdvisoryKnowledgeBase

# Initialize FastAPI Application
app = FastAPI(
    title="AI-Integrated Crop Disease Advisory API",
    description=(
        "RESTful API service for foliar crop disease diagnosis and agronomic advisory. "
        "Capstone Project - Samsung Innovation Campus (SIC) - Group 5."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware for Frontend Integration (Ma Hong Hai)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate engines
inference_engine = CropDiseaseInferenceEngine()
advisory_engine = AdvisoryKnowledgeBase()


# Response Schemas
class HealthCheckResponse(BaseModel):
    status: str
    service: str
    version: str
    model_architecture: str
    weights_loaded: bool
    num_supported_classes: int
    confidence_threshold: float
    target_latency: str


@app.get("/", tags=["Root"])
def root_endpoint() -> Dict[str, Any]:
    """Root endpoint welcoming users and providing documentation links."""
    return {
        "project": "AI-Integrated Crop Disease Advisory and Image-Based Diagnosis System",
        "program": "Samsung Innovation Campus (SIC) AI Course",
        "team": "Group 5",
        "backend_lead": "Nguyen Quoc Khanh",
        "status": "Online",
        "documentation": "/docs",
        "health_check": "/api/health",
        "diagnose_endpoint": "/api/diagnose [POST]",
    }


@app.get("/api/health", response_model=HealthCheckResponse, tags=["System"])
def health_check() -> HealthCheckResponse:
    """Verifies backend API service, model engine, and knowledge base availability."""
    return HealthCheckResponse(
        status="HEALTHY",
        service="SIC Crop Disease Advisory API",
        version="1.0.0",
        model_architecture="MobileNetV2 (224x224 RGB)",
        weights_loaded=inference_engine.weights_loaded,
        num_supported_classes=len(CLASS_NAMES),
        confidence_threshold=advisory_engine.threshold,
        target_latency="< 200 ms",
    )


@app.post("/api/diagnose", tags=["Diagnosis"])
async def diagnose_leaf_image(
    file: UploadFile = File(..., description="Uploaded leaf photograph (JPEG or PNG)"),
    custom_threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Override confidence cutoff threshold"
    )
) -> JSONResponse:
    """
    Main Diagnosis Endpoint:
    1. Validates image format, resolution, and corruption.
    2. Runs MobileNetV2 inference and computes calibrated confidence.
    3. Queries Agricultural Advisory Engine through confidence-gated logic.
    4. Delivers consolidated JSON payload with treatments, safety disclaimers, and latency.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không tìm thấy tệp ảnh tải lên."
        )

    try:
        # Read file bytes
        file_bytes = await file.read()

        # Step 1: Input Validation
        validate_image_file(file_bytes, file.filename or "uploaded_image.jpg")

        # Step 2: Model Inference
        inference_result = inference_engine.predict(file_bytes)

        # Step 3: Confidence-Gated Decision Engine
        if custom_threshold is not None:
            original_threshold = advisory_engine.threshold
            advisory_engine.threshold = custom_threshold
            payload = advisory_engine.process_diagnosis(inference_result)
            advisory_engine.threshold = original_threshold
        else:
            payload = advisory_engine.process_diagnosis(inference_result)

        # Include file metadata
        payload["image_metadata"] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(file_bytes),
        }

        return JSONResponse(status_code=status.HTTP_200_OK, content=payload)

    except InputValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý nội bộ backend: {str(e)}"
        )


@app.get("/api/diseases", tags=["Knowledge Base"])
def get_all_diseases() -> Dict[str, Any]:
    """Retrieves full catalog of all crop pathologies in the advisory database."""
    diseases = advisory_engine.list_all_diseases()
    return {
        "total": len(diseases),
        "diseases": diseases
    }


@app.get("/api/diseases/{disease_id}", tags=["Knowledge Base"])
def get_disease_detail(disease_id: str) -> Dict[str, Any]:
    """Retrieves full treatment, agronomic steps, and safety precautions for a specific disease ID."""
    record = advisory_engine.get_disease_by_id(disease_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy thông tin bệnh với mã: '{disease_id}'. Vui lòng kiểm tra lại."
        )
    return {
        "disease_id": disease_id,
        "advisory_data": record,
        "disclaimer": advisory_engine.disclaimer,
    }


@app.get("/api/crops", tags=["Knowledge Base"])
def get_supported_crops() -> Dict[str, Any]:
    """Lists target crops covered under the capstone project."""
    return {
        "supported_crops": [
            {
                "crop_name": "Cà chua (Tomato)",
                "conditions": [
                    "Tomato_healthy",
                    "Tomato_Early_blight",
                    "Tomato_Late_blight",
                    "Tomato_Septoria_leaf_spot"
                ]
            },
            {
                "crop_name": "Khoai tây (Potato)",
                "conditions": [
                    "Potato_healthy",
                    "Potato_Early_blight",
                    "Potato_Late_blight"
                ]
            },
            {
                "crop_name": "Ngô (Corn)",
                "conditions": [
                    "Corn_healthy",
                    "Corn_Common_rust",
                    "Corn_Northern_Leaf_Blight",
                    "Corn_Gray_leaf_spot"
                ]
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
