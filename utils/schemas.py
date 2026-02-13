from pydantic import BaseModel, Field
from typing import List


class SummarySchema(BaseModel):
    overview: str = Field(description="Concise overview of the document")
    key_points: List[str] = Field(description="Main key points")
    word_count: int = Field(description="Total word count of summary")