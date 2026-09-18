from fastapi import FastAPI, File, HTTPException, UploadFile, status

from multimodal_diagnosis.config import get_settings
from multimodal_diagnosis.models.xray import XrayModel
from multimodal_diagnosis.schemas import XrayDiagnosisResponse
from multimodal_diagnosis.services.diagnosis import InvalidImageError, XrayDiagnosisService

app = FastAPI(
    title="Multimodal Diagnosis API",
    description="X-ray-first research API with a future clinical-notes extension point.",
    version="0.1.0",
)

settings = get_settings()
xray_service = XrayDiagnosisService(
    XrayModel(settings.xray_model_weights, settings.xray_checkpoint), settings
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/diagnose/xray",
    response_model=XrayDiagnosisResponse,
    status_code=status.HTTP_200_OK,
)
async def diagnose_xray(image: UploadFile = File(...)) -> XrayDiagnosisResponse:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must have an image content type")
    image_bytes = await image.read(settings.max_upload_bytes + 1)
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the maximum upload size")
    try:
        return xray_service.diagnose(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
