from langgraph.graph import StateGraph, START, END
from graph.state import GraphState
from graph.nodes import check_length_node, direct_summarize_node, generate_structured_summary, evaluate_confidence, combine_summaries_node, retry_node, route_after_check, route_after_confidence, summarize_chunks_node


def build_graph():
    workflow = StateGraph(GraphState)
    # Step 1: Add All Nodes
    workflow.add_node("check_length", check_length_node)
    workflow.add_node("direct", direct_summarize_node)
    workflow.add_node("chunk", summarize_chunks_node)
    workflow.add_node("combine", combine_summaries_node)

    workflow.add_node("generate_structured_summary", generate_structured_summary)
    workflow.add_node("evaluate_confidence", evaluate_confidence)
    workflow.add_node("retry_node", retry_node)

    # Step 2: START → Length Check
    workflow.add_edge(START, "check_length")

    # Step 3: Conditional Routing
    workflow.add_conditional_edges(
        "check_length",
        route_after_check,
        {
            "direct": "direct",
            "chunk": "chunk"
        }
    )

    # Step 4: Direct Path
    workflow.add_edge("direct", "generate_structured_summary")

    # Step 5: Chunk Path
    workflow.add_edge("chunk", "combine")
    workflow.add_edge("combine", "generate_structured_summary")

    # Step 6: Structured Summary → Evaluate
    workflow.add_edge("generate_structured_summary", "evaluate_confidence")

    # Step 7: Confidence-Based Retry
    workflow.add_conditional_edges(
        "evaluate_confidence",
        route_after_confidence,
        {
            "retry": "retry_node",
            "end": END
        }
    )

    # Retry loops back to regenerate summary
    workflow.add_edge("retry_node", "generate_structured_summary")

    #Compile Graph
    return workflow.compile()