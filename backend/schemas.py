from pydantic import BaseModel, Field
from typing import Literal

class WebcamRequest(BaseModel):
    camera_index: int = Field(default=0, ge=0, le=10)

class Zone(BaseModel):
    id: str
    name: str
    severity: Literal["low", "medium", "high", "critical"] = "high"
    points: list[list[float]]

class ZonesRequest(BaseModel):
    zones: list[Zone]
