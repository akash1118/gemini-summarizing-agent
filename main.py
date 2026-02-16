import gradio as gr
from graph.workflow import build_graph
from utils.pdf_loader import extract_text_from_pdf
import uuid

graph = build_graph()
session_id = str(uuid.uuid4())

def toggle_input(mode):
    """
    Show PDF uploader if mode == PDF,
    Show text box if mode == Query.
    """
    if mode == "PDF":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


def run_agent(mode, pdf_file, query_text):
    # -------------------------
    # 1️⃣ Get Input Based on Mode
    # -------------------------

    if mode == "PDF":
        if pdf_file is None:
            return "Please upload a PDF file."

        text = extract_text_from_pdf(pdf_file.name)

        if not text:
            return "Could not extract text from PDF."

    else:  # Query mode
        if not query_text or not query_text.strip():
            return "Please enter a query."

        text = query_text.strip()

    # -------------------------
    # 2️⃣ Initialize Graph State
    # -------------------------

    result = graph.invoke(
    {"input_text": text},
    config={"configurable": {"thread_id": session_id}}
)


    # -------------------------
    # 3️⃣ Format Output
    # -------------------------

    summary = result.get("summary", {})

    if isinstance(summary, dict):
        overview = summary.get("overview", "")
        key_points = summary.get("key_points", [])
        word_count = summary.get("word_count", 0)

        formatted = f"### Overview\n{overview}\n\n### Key Points\n"
        for point in key_points:
            formatted += f"- {point}\n"

        formatted += f"\nWord Count: {word_count}"
        formatted += f"\nConfidence Score: {result.get('confidence_score', 0)}"

        return formatted

    return summary


# -------------------------
# 🎨 Gradio UI
# -------------------------

with gr.Blocks() as demo:
    gr.Markdown("# 📄🔎 Smart Summarizer (LangGraph V4)")

    mode_selector = gr.Dropdown(
        ["PDF", "Query"],
        label="Select Input Type",
        value="PDF"
    )

    pdf_input = gr.File(
        file_types=[".pdf"],
        label="Upload PDF",
        visible=True
    )

    query_input = gr.Textbox(
        label="Enter Topic / Query",
        lines=3,
        visible=False
    )

    summarize_btn = gr.Button("Summarize")

    output_box = gr.Textbox(
        label="Summary Output",
        lines=20
    )

    # Toggle visibility when dropdown changes
    mode_selector.change(
        toggle_input,
        inputs=mode_selector,
        outputs=[pdf_input, query_input]
    )

    summarize_btn.click(
        run_agent,
        inputs=[mode_selector, pdf_input, query_input],
        outputs=output_box
    )


if __name__ == "__main__":
    demo.launch()
