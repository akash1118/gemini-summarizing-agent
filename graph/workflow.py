from langgraph.graph import StateGraph, START, END
from graph.state import GraphState
from graph.nodes import check_length_node, direct_summarize_node, generate_structured_summary, evaluate_confidence, combine_summaries_node, retry_node, route_after_check, route_after_confidence, summarize_chunks_node, detect_input_type_node, detect_summary_type_node, summarize_search_node, web_search_node, route_after_input_detection
from langgraph.checkpoint.memory import MemorySaver


def build_graph():
    workflow = StateGraph(GraphState)

    # -----------------------------
    # 1️⃣ Add Nodes
    # -----------------------------

    workflow.add_node("detect_input_type", detect_input_type_node)
    workflow.add_node("detect_summary_type", detect_summary_type_node)

    workflow.add_node("check_length", check_length_node)
    workflow.add_node("direct", direct_summarize_node)
    workflow.add_node("chunk", summarize_chunks_node)
    workflow.add_node("combine", combine_summaries_node)

    workflow.add_node("web_search", web_search_node)
    workflow.add_node("summarize_search", summarize_search_node)

    workflow.add_node("generate_structured_summary", generate_structured_summary)
    workflow.add_node("evaluate_confidence", evaluate_confidence)
    workflow.add_node("retry_node", retry_node)

    # -----------------------------
    # 2️⃣ START → Detect Input
    # -----------------------------

    workflow.add_edge(START, "detect_input_type")

    # Detect summary style early
    workflow.add_edge("detect_input_type", "detect_summary_type")

    # -----------------------------
    # 3️⃣ Route Topic vs Document
    # -----------------------------

    workflow.add_conditional_edges(
        "detect_summary_type",
        route_after_input_detection,
        {
            "search": "web_search",
            "document": "check_length"
        }
    )

    # -----------------------------
    # 4️⃣ Topic Path (Web Search)
    # -----------------------------

    workflow.add_edge("web_search", "summarize_search")
    workflow.add_edge("summarize_search", "evaluate_confidence")

    # -----------------------------
    # 5️⃣ Document Path
    # -----------------------------

    workflow.add_conditional_edges(
        "check_length",
        route_after_check,
        {
            "direct": "direct",
            "chunk": "chunk"
        }
    )

    workflow.add_edge("direct", "generate_structured_summary")

    workflow.add_edge("chunk", "combine")
    workflow.add_edge("combine", "generate_structured_summary")

    workflow.add_edge("generate_structured_summary", "evaluate_confidence")

    # -----------------------------
    # 6️⃣ Retry Loop
    # -----------------------------

    workflow.add_conditional_edges(
        "evaluate_confidence",
        route_after_confidence,
        {
            "retry": "retry_node",
            "end": END
        }
    )

    workflow.add_edge("retry_node", "generate_structured_summary")

    # -----------------------------
    # 7️⃣ Compile
    # -----------------------------
    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)
