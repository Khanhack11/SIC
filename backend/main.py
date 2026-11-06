"""
FastAPI RESTful Service for Crop Disease Advisory System
Author: Nguyen Quoc Khanh (Backend Developer & Database Architect)
SIC Capstone Project - Group 5
"""

import time
import logging
import os
import torch
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import anyio
from contextlib import asynccontextmanager
from functools import partial
from threading import BoundedSemaphore
from pydantic import BaseModel

from backend.config import (CONFIDENCE_THRESHOLD, CLASS_NAMES, TORCH_NUM_THREADS, INFERENCE_RUNTIME, INFERENCE_DEVICE,
                            INFERENCE_CONCURRENCY, MAX_FILE_SIZE_BYTES, SAMPLE_IMAGES_DIR, FRONTEND_DIR)
from backend.inference_engine import (
    CropDiseaseInferenceEngine,
    validate_image_file,
    InputValidationError,
    ModelNotReadyError,
    crop_upload,
)
from backend.advisory_engine import AdvisoryKnowledgeBase
from backend.leaf_detector import detect_leaf_regions, auto_crop_leaf
from backend.chatbot_engine import AgronomyChatbot
from backend.weather_advisory import get_agro_weather_advisory, REGIONS_CATALOG

# Initialize FastAPI Application
@asynccontextmanager
async def lifespan(application):
    application.state.inference_limiter = anyio.CapacityLimiter(INFERENCE_CONCURRENCY)
    try:
        yield
    finally:
        del application.state.inference_limiter


app = FastAPI(
    title="AI-Integrated Crop Disease Advisory API",
    description=(
        "RESTful API service for foliar crop disease diagnosis and agronomic advisory. "
        "Capstone Project - Samsung Innovation Campus (SIC) - Group 5."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
torch.set_num_threads(TORCH_NUM_THREADS)
field_model_path = os.environ.get('SIC_FIELD_MODEL_PATH')
if field_model_path:
    from backend.field_inference import FieldInferenceEngine
    inference_engine = FieldInferenceEngine(field_model_path, onnx_path=os.environ.get('SIC_FIELD_ONNX_PATH'), device=INFERENCE_DEVICE)
else:
    inference_engine = CropDiseaseInferenceEngine(runtime=INFERENCE_RUNTIME, device=INFERENCE_DEVICE)
advisory_engine = AdvisoryKnowledgeBase()
chatbot = AgronomyChatbot(advisory_engine)
inference_slots = BoundedSemaphore(INFERENCE_CONCURRENCY)
app.mount("/demo", StaticFiles(directory=SAMPLE_IMAGES_DIR / "demo", check_dir=False), name="demo")


@app.get("/app", include_in_schema=False)
@app.get("/frontend_demo.html", include_in_schema=False)
def frontend():
    return FileResponse(FRONTEND_DIR / "frontend_demo.html")


@app.get("/frontend.js", include_in_schema=False)
def frontend_script():
    return FileResponse(FRONTEND_DIR / "frontend.js", media_type="text/javascript")


@app.get('/crop.js', include_in_schema=False)
def crop_script():
    return FileResponse(FRONTEND_DIR / 'crop.js', media_type='text/javascript')


def process_image(file_bytes, filename, threshold, crop_box=None, visible_damage=False, field_photo=True):
    """Run CPU work off the event loop with request-local advisory settings."""
    with inference_slots:
        validate_image_file(file_bytes, filename)
        inference_result = inference_engine.predict(crop_upload(file_bytes,crop_box))
        inference_result['visible_damage'] = visible_damage
        inference_result['field_photo'] = field_photo
        return advisory_engine.process_diagnosis(inference_result, threshold=threshold)


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
    inference_runtime: str
    device: str
    calibrated: bool


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
        "frontend": "/app",
    }


@app.get("/api/health", response_model=HealthCheckResponse, tags=["System"])
def health_check() -> HealthCheckResponse:
    """Verifies backend API service, model engine, and knowledge base availability."""
    return HealthCheckResponse(
        status="HEALTHY" if inference_engine.weights_loaded else "MODEL_NOT_READY",
        service="SIC Crop Disease Advisory API",
        version="1.0.0",
        model_architecture="MobileNetV3-Large field (224x224 RGB letterbox)" if field_model_path else "MobileNetV2 (224x224 RGB)",
        weights_loaded=inference_engine.weights_loaded,
        num_supported_classes=len(CLASS_NAMES),
        confidence_threshold=advisory_engine.threshold,
        target_latency="< 200 ms",
        inference_runtime=inference_engine.runtime,
        device=str(inference_engine.device),
        calibrated=inference_engine.calibrated,
    )


@app.post("/api/diagnose", tags=["Diagnosis"])
async def diagnose_leaf_image(
    file: UploadFile = File(..., description="Uploaded leaf photograph (JPEG or PNG)"),
    crop_box: Optional[str] = Query(None, description="Optional normalized x,y,width,height crop, each in [0,1]"),
    visible_damage: bool = Query(False, description="User observes holes, spots or deformation; healthy predictions require review"),
    field_photo: bool = Query(True, description="Default for user photographs; false is reserved for controlled validation examples"),
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
        processing_started = time.perf_counter()
        # Read file bytes
        file_bytes = await file.read(MAX_FILE_SIZE_BYTES + 1)
        payload = await anyio.to_thread.run_sync(partial(process_image, file_bytes,
                                                       file.filename or "uploaded_image.jpg", custom_threshold, crop_box, visible_damage, field_photo),
                                                limiter=getattr(app.state, "inference_limiter", None))

        # Include file metadata
        payload["image_metadata"] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(file_bytes),
        }

        processing_ms = (time.perf_counter() - processing_started) * 1000
        return JSONResponse(status_code=status.HTTP_200_OK, content=payload,
                            headers={"Server-Timing": f"inference;dur={payload['prediction']['latency_ms']:.2f}, processing;dur={processing_ms:.2f}"})

    except ModelNotReadyError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except InputValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logging.getLogger(__name__).exception("Diagnosis processing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể xử lý ảnh lúc này. Vui lòng thử lại hoặc chọn ảnh khác."
        )
    finally:
        await file.close()


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




# Agronomy & Chatbot Request Models
class WeatherRiskRequest(BaseModel):
    temperature_c: float
    humidity_pct: float
    crop: Optional[str] = None


class ChatMessageRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


@app.get("/api/diseases/{disease_id}/agronomy", tags=["Knowledge Base"])
def get_disease_agronomy(disease_id: str) -> Dict[str, Any]:
    """Retrieves full agronomic profile including soil, climate, pest vectors, and exact chemical dosages."""
    record = advisory_engine.get_agronomy_detail(disease_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy hồ sơ nông học cho mã: '{disease_id}'"
        )
    return {
        "disease_id": disease_id,
        "agronomy_data": record,
        "disclaimer": advisory_engine.disclaimer
    }


@app.post("/api/weather-advisory", tags=["Agricultural Advisory"])
def weather_advisory(req: WeatherRiskRequest) -> Dict[str, Any]:
    """Evaluates environmental telemetry (temperature, humidity) against epidemic triggers."""
    alerts = advisory_engine.assess_weather_risk(req.temperature_c, req.humidity_pct, req.crop)
    return {
        "status": "ALERT" if alerts else "SAFE",
        "telemetry": {
            "temperature_c": req.temperature_c,
            "humidity_pct": req.humidity_pct,
            "crop_filter": req.crop or "Tất cả cây trồng"
        },
        "total_risks_detected": len(alerts),
        "alerts": alerts,
        "general_advice": (
            "Điều kiện thời tiết đang ở ngưỡng nguy cơ cao bùng phát dịch bệnh. Khuyến cáo kiểm tra đồng ruộng và chuẩn bị thuốc BVTV phòng trị kịp thời."
            if alerts else "Điều kiện thời tiết hiện tại trong ngưỡng an toàn, ít nguy cơ bùng phát dịch hại lớn."
        )
    }


@app.post("/api/chat", tags=["Agricultural Advisory"])
def farm_chat_assistant(req: ChatMessageRequest) -> Dict[str, Any]:
    """AI Agricultural Chatbot Assistant powered by Google Gemini API and SIC Knowledge Base."""
    if not req.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Vui lòng nhập câu hỏi.")
    return chatbot.chat(req.message, getattr(req, "context", None))


@app.get("/api/dataset-insights", tags=["Knowledge Base"])
def get_dataset_insights() -> Dict[str, Any]:
    """Returns dataset survey comparing Kaggle New Plant Diseases Dataset vs. Real-World Field datasets."""
    kb_data = advisory_engine.data
    return {
        "kaggle_benchmark": kb_data.get("kaggle_reference_dataset", {}),
        "our_field_v2_dataset": {
            "total_records": 20902,
            "training_samples": 16225,
            "validation_samples": 2740,
            "test_samples": 1937,
            "single_leaf_crops_extracted": 3160,
            "international_data_sources": [
                "PlantVillage (Lab standard)",
                "PlantDoc (Real-world in-the-wild field conditions)",
                "CCMT (Crop-Care Multi-Country)",
                "BARI (Bangladesh Agricultural Research Institute)",
                "Holeta (Ethiopia Potato Late Blight)",
                "IIT Kharagpur (India Potato Leaf Dataset)",
                "Indonesia Potato Field Survey",
                "Pakistan Tomato Camera Captures"
            ],
            "advantages": [
                "Không bị overfitting phông nền phòng thí nghiệm",
                "Hỗ trợ crop bounding-box lá đơn ngoài thực địa",
                "Tăng trưởng F1 thực địa +39.03 pp trên PlantDoc"
            ]
        }
    }


@app.post("/api/detect-leaves", tags=["Diagnosis"])
async def detect_leaves_endpoint(
    file: UploadFile = File(..., description="Uploaded leaf photograph to detect bounding boxes")
) -> JSONResponse:
    """
    Automatic Leaf & Disease Lesion Detection:
    Segments plant foliage, extracts candidate bounding boxes around leaves and lesions,
    and returns annotated image preview with magenta bounding boxes.
    """
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Không tìm thấy tệp ảnh tải lên.")
    file_bytes = await file.read(MAX_FILE_SIZE_BYTES + 1)
    validate_image_file(file_bytes, file.filename or "uploaded.jpg")
    result = await anyio.to_thread.run_sync(detect_leaf_regions, file_bytes)
    return JSONResponse(content=result)


@app.post("/api/auto-diagnose", tags=["Diagnosis"])
async def auto_diagnose_leaf(
    file: UploadFile = File(..., description="Uploaded leaf photograph for auto-detection and diagnosis"),
    visible_damage: bool = Query(False, description="User observed leaf damage"),
    custom_threshold: Optional[float] = Query(None, ge=0.0, le=1.0)
) -> JSONResponse:
    """
    End-to-End Automatic Detection and Diagnosis:
    1. Detects all leaves & lesions with bounding box proposals.
    2. Crops the primary diseased leaf.
    3. Runs diagnosis and advisory inference.
    4. Returns diagnosis payload bundled with detection bounding boxes & preview.
    """
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Không tìm thấy tệp ảnh tải lên.")
    file_bytes = await file.read(MAX_FILE_SIZE_BYTES + 1)
    validate_image_file(file_bytes, file.filename or "uploaded.jpg")

    detection = await anyio.to_thread.run_sync(detect_leaf_regions, file_bytes)
    primary_crop = detection.get("primary_crop_str")

    payload = await anyio.to_thread.run_sync(
        partial(process_image, file_bytes, file.filename or "uploaded_image.jpg",
                custom_threshold, primary_crop, visible_damage, True),
        limiter=getattr(app.state, "inference_limiter", None)
    )
    payload["detection_info"] = detection
    return JSONResponse(content=payload)


# =====================================================================
# Pillar 3: Regional Agro-Meteorological Epidemic Early Warning
# =====================================================================

@app.get("/api/weather/regions", tags=["Agro Weather & Early Warning"])
def get_weather_regions() -> Dict[str, Any]:
    """Lists supported agricultural regions for meteorological and disease risk monitoring."""
    return {
        "regions": [{"id": r["id"], "name": r["name"], "crops": r["crops"]} for r in REGIONS_CATALOG]
    }


@app.get("/api/weather/advisory", tags=["Agro Weather & Early Warning"])
def get_weather_risk_advisory(region: str = Query("thai_nguyen", description="Region ID")) -> Dict[str, Any]:
    """
    Provides real-time agro-meteorological monitoring and calculates
    Epidemiological Disease Index (EDI) risks for Late Blight, Early Blight, Rust, etc.
    """
    return get_agro_weather_advisory(region)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
