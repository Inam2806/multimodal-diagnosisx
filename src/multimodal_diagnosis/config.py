from functools import lru_cache
import os


class Settings:
    """Runtime settings loaded from environment variables."""

    xray_model_weights: str
    xray_checkpoint: str
    xray_top_k: int
    xray_threshold: float
    max_upload_bytes: int

    def __init__(self) -> None:
        self.xray_model_weights = os.getenv(
            "XRAY_MODEL_WEIGHTS", "densenet121-res224-all"
        )
        self.xray_checkpoint = os.getenv("XRAY_CHECKPOINT", "")
        self.xray_top_k = int(os.getenv("XRAY_TOP_K", "5"))
        self.xray_threshold = float(os.getenv("XRAY_THRESHOLD", "0.5"))
        self.max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
        if not 0 < self.xray_threshold < 1:
            raise ValueError("XRAY_THRESHOLD must be between 0 and 1")
        if self.xray_top_k < 1:
            raise ValueError("XRAY_TOP_K must be at least 1")
        if self.max_upload_bytes < 1:
            raise ValueError("MAX_UPLOAD_BYTES must be positive")


@lru_cache
def get_settings() -> Settings:
    return Settings()
