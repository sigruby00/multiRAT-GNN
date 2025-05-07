from pydantic_settings import BaseSettings
from datetime import datetime
from typing import Any

class RecordData(BaseSettings):
    timestamp: datetime
    data: Any

class RobotRecordData(BaseSettings):
    timestamp: datetime
    robot_id: Any
    features: Any
    target: Any
