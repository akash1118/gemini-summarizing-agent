from urllib3.util import retry
from services.llm_service import get_llm, get_structured_llm
from graph.state import GraphState
from utils.prompt import CONFIDENCE_PROMPT, STRUCTURED_SUMMARY_PROMPT, SUMMARY_PROMPT, COMBINE_PROMPT
import json

llm = get_llm()
structured_llm =get_structured_llm()

MAX_CHARS = 4000
MAX_RETRIES = 2
CONFIDENCE_THRESHOLD = 0.75


def check_length_node(state: GraphState):
    """
    Node: Check length of input text.
    """
    text = state["input_text"]
    print(f"length of text---------->{len(text)}")
    if len(text) < MAX_CHARS:
        print("======Chunks not splitted========")
        return { "chunks": [], "partial_summaries": []}

    # Slipt into chunks
    print("======Splitting chunks========")
    chunks = [
        text[i:i + MAX_CHARS]
        for i in range(0, len(text), MAX_CHARS)
    ]
    print(f"chunks ========> {chunks}")
    return {"chunks": chunks, "partial_summaries": []}

def route_after_check(state: GraphState):
    print(f"Checking state==========>{state}")
    if not state["chunks"]:
        return "direct"
    return "chunk"

def direct_summarize_node(state: GraphState):
    """
    Node: Direct summarize (short text)
    """
    print(f"======Sumarizing Directly======")
    text = state["input_text"]
    prompt = SUMMARY_PROMPT.format(text=text)
    response = llm.invoke(prompt)
    return {"summary": response.content}

def summarize_chunks_node(state: GraphState):
    """
    Node: summarize each chunk.
    """
    summaries = []
    print(f"======Sumarizing Chunks======")
    for chunk in state["chunks"]:
        prompt = f"Summarize clearly:\n{chunk}"
        response = llm.invoke(prompt)
        summaries.append(response.content)

        return {"partial_summaries": summaries}

def combine_summaries_node(state: GraphState):
    """
    Node: Combine summaries.
    """
    print(f"Partial State ==========> {state['partial_summaries']}")
    combined_text = "\n".join(state["partial_summaries"])
    prompt = COMBINE_PROMPT.format(combined_text=combined_text)
    response = llm.invoke(prompt)

    return { "summary": response.content}


def generate_structured_summary(state: GraphState):
    text = state["input_text"]

    print(f"text===========>{text}")
    prompt = STRUCTURED_SUMMARY_PROMPT.format(text=text)

    response = structured_llm.invoke(prompt)

    try:
        parsed = response.__dict__
    except Exception:
        # fallback if model outputs invalid json
        parsed = {
            "overview": response.content,
            "key_points": [],
            "word_count": len(response.content.split())
        }
        return {"summary": parsed}
    

def evaluate_confidence(state):
    summary = state["summary"]

    prompt = CONFIDENCE_PROMPT.format(summary=summary)

    response = llm.invoke(prompt)

    try:
        score = float(response.content.strip())
    except:
        # default fallback score
        score = 0.5

    return {"confidence_score": score}


def route_after_confidence(state: GraphState):
    if (
        state["confidence_score"] < CONFIDENCE_THRESHOLD
        and state["retry_count"]< MAX_RETRIES
    ):
    
        return "retry"

    return "end"

def retry_node(state: GraphState):
    return {
        "retry_count": state["retry_count"] + 1
    }