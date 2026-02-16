from typing import TypedDict, List

class GraphState(TypedDict):
    input_text: str
    input_type: str
    search_results: str
    summary_type: str
    chunks: List[str]
    partial_summaries: List[str]
    summary: str
    confidence_score: float
    retry_count: int
    history: List[dict]
