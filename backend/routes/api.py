"""API routes for Virtual Try-On AI."""

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

from backend.utils.image_processor import ImageProcessor
from backend.utils.inference import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["virtual-try-on"])
processor = ImageProcessor()


class GarmentClassificationRequest(BaseModel):
    """Request model for garment classification."""

    image_base64: str


class GarmentClassificationResponse(BaseModel):
    """Response model for garment classification."""

    garment_type: str
    confidence: float
    alternatives: list[str]


class VirtualTryOnRequest(BaseModel):
    """Request model for virtual try-on."""

    person_image_base64: str
    garment_image_base64: str
    garment_type: Optional[str] = None


class VirtualTryOnResponse(BaseModel):
    """Response model for virtual try-on."""

    result_image_base64: str
    garment_detected: str
    confidence: float
    processing_time: float


@router.post("/classify-garment", response_model=GarmentClassificationResponse)
async def classify_garment(request: GarmentClassificationRequest):
    """
    Classify clothing type from an image.

    Args:
        request: Base64 encoded image

    Returns:
        Classification results
    """
    try:
        pipeline = get_pipeline()

        # Decode base64 image
        image = processor.base64_to_image(request.image_base64)

        # Classify
        result = pipeline.classify_garment(image)

        return GarmentClassificationResponse(
            garment_type=result["garment_type"],
            confidence=result["confidence"],
            alternatives=result["alternatives"],
        )

    except Exception as e:
        logger.error(f"Error in garment classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/virtual-tryon", response_model=VirtualTryOnResponse)
async def virtual_tryon(request: VirtualTryOnRequest):
    """
    Generate virtual try-on result.

    Args:
        request: Images and optional garment type

    Returns:
        Try-on result with generated image
    """
    try:
        pipeline = get_pipeline()

        # Decode base64 images
        person_image = processor.base64_to_image(request.person_image_base64)
        garment_image = processor.base64_to_image(request.garment_image_base64)

        # Generate try-on
        result = pipeline.generate_try_on(
            person_image=person_image,
            garment_image=garment_image,
            garment_type=request.garment_type,
        )

        # Encode result image to base64
        result_base64 = processor.image_to_base64(result["result_image"])

        return VirtualTryOnResponse(
            result_image_base64=result_base64,
            garment_detected=result["garment_detected"],
            confidence=result["confidence"],
            processing_time=result["processing_time"],
        )

    except Exception as e:
        logger.error(f"Error in virtual try-on: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/virtual-tryon-upload")
async def virtual_tryon_upload(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    garment_type: Optional[str] = Form(None),
):
    """
    Generate virtual try-on with file uploads.

    Args:
        person_image: Uploaded person image
        garment_image: Uploaded garment image
        garment_type: Optional garment type

    Returns:
        Try-on result
    """
    try:
        from PIL import Image
        from io import BytesIO

        pipeline = get_pipeline()

        # Read uploaded files
        person_image_data = await person_image.read()
        garment_image_data = await garment_image.read()

        # Convert to PIL Images
        person_pil = Image.open(BytesIO(person_image_data)).convert("RGB")
        garment_pil = Image.open(BytesIO(garment_image_data)).convert("RGB")

        # Generate try-on
        result = pipeline.generate_try_on(
            person_image=person_pil,
            garment_image=garment_pil,
            garment_type=garment_type,
        )

        # Encode result image to base64
        result_base64 = processor.image_to_base64(result["result_image"])

        return VirtualTryOnResponse(
            result_image_base64=result_base64,
            garment_detected=result["garment_detected"],
            confidence=result["confidence"],
            processing_time=result["processing_time"],
        )

    except Exception as e:
        logger.error(f"Error in virtual try-on upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Virtual Try-On AI"}


@router.post("/clear-cache")
async def clear_cache():
    """Clear GPU/CPU cache."""
    try:
        pipeline = get_pipeline()
        pipeline.clear_cache()
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
