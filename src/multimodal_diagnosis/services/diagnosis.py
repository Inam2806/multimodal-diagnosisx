from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from multimodal_diagnosis.config import Settings
from multimodal_diagnosis.models.xray import XrayModel
from multimodal_diagnosis.schemas import XrayDiagnosisResponse, XrayPrediction


class InvalidImageError(ValueError):
    """Raised when an upload cannot be decoded as an image."""


class XrayDiagnosisService:
    def __init__(self, model: XrayModel, settings: Settings) -> None:
        self.model = model
        self.settings = settings

    def diagnose(self, image_bytes: bytes) -> XrayDiagnosisResponse:
        image = self._preprocess(image_bytes)
        scores = self.model.predict(image)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        predictions = [
            XrayPrediction(
                label=label,
                probability=round(probability, 6),
                positive=probability >= self.settings.xray_threshold,
            )
            for label, probability in ranked[: self.settings.xray_top_k]
        ]
        return XrayDiagnosisResponse(
            model=self.model.display_name,
            predictions=predictions,
        )

    @staticmethod
    def _preprocess(image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                grayscale = image.convert("L").resize((224, 224))
                array = np.asarray(grayscale, dtype=np.float32)
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("The uploaded file is not a readable image") from exc
        return (array / 255.0) * 2.0 - 1.0
