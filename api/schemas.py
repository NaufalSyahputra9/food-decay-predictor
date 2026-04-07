# api/schemas.py
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional

# --- Input Schemas ---
class FoodSessionCreate(BaseModel):
    food_label: str = Field(..., description="Label makanan dari hasil OSR/CLIP")
    storage_type: str = Field(default="room_temp")

class SnapshotUpload(BaseModel):
    session_id: str
    day_index: int = Field(..., description="0 = hari pertama")

# --- Output Schemas ---
class PredictionResult(BaseModel):
    session_id: str
    days_remaining: float
    freshness_status: str
    confidence: float
    decay_curve: Optional[List[float]] = None

    model_config = ConfigDict(from_attributes=True)