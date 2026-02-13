from services.llm_service import get_llm
from graph.state import GraphState
from utils.prompt import SUMMARY_PROMPT, COMBINE_PROMPT

llm = get_llm()
MAX_CHARS = 4000



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

def direct_summarize_node(state: GraphState):
    """
    Node: Direct summarize (short text)
    """
    print(f"======Sumarizing Directly======")
    text = state["input_text"]
    prompt = SUMMARY_PROMPT.format(text=text)
    response = llm.invoke(prompt)
    return {"summary": response.content}

def summarize_chunk_node(state: GraphState):
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
