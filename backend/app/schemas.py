from pydantic import BaseModel
from typing import Optional, Dict

class UserCreate(BaseModel):
    name: str
    email: str
    birth_year: Optional[int]
    gender: Optional[str]
    health_constraints: Optional[Dict] = {}   # e.g. {"heart_condition": True, "injury":"knee"}
    consent_processed: bool = True

class RecommendRequest(BaseModel):
    user_id: str
    activity: str
    scheduled_duration_min: int
    candidates: Optional[list] = None  # list of start times (minutes)
    sleep_hours: Optional[float] = None
    mood: Optional[int] = None
