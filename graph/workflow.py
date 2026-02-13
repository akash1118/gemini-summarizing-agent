from langgraph.graph import StateGraph, START, END
from graph.state import GraphState
from graph.nodes import check_length_node, direct_summarize_node, summarize_chunk_node, combine_summaries_node

def route_after_check(state: GraphState):
    print(f"Checking state==========>{state}")
    if not state["chunks"]:
        return "direct"
    return "chunk"



def build_graph():
    workflow = StateGraph(GraphState)
    # Add node
    workflow.add_node("check_length", check_length_node)
    workflow.add_node("direct", direct_summarize_node)
    workflow.add_node("chunk", summarize_chunk_node)
    workflow.add_node("combine", combine_summaries_node)

    # Edges
    workflow.add_edge(START, "check_length")

    workflow.add_conditional_edges(
        "check_length",
        route_after_check,
        {
            "direct": "direct",
            "chunk": "chunk"
        }
    )

    # Define flow
    workflow.add_edge("direct", END)
    workflow.add_edge("chunk", "combine")
    workflow.add_edge("combine", END)

    #Compile Graph
    return workflow.compile()