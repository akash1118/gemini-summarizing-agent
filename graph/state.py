from typing import TypedDict, List

class GraphState(TypedDict):
    input_text: str
    chunks: List[str]
    partial_summaries: List[str]
    summary: str
    confidence_score: float
    retry_count: int


