import gradio as gr
from graph.workflow import build_graph
from utils.pdf_loader import extract_text_from_pdf

graph = build_graph()

def summarize_pdf(file):
    if file is None:
        return "Please Upload a file."
    
    # Extract text
    text = extract_text_from_pdf(file)

    if not text:
        return "Could not extract from PDF."
    
    # Call LangGraph
    result = graph.invoke({
        "input_text": text,
        "summary": ""
    })
    return result["summary"]


with gr.Blocks() as demo:
    gr.Markdown("# PDF Summarizer (Langgraph Agent)")

    with gr.Row():
        pdf_input = gr.File(file_types=[".pdf"], label = "Upload PDF")

    summarize_btn = gr.Button("Summarize")

    output_box = gr.Textbox(
        label = "Summary",
        lines=15
    )

    summarize_btn.click(
        summarize_pdf,
        inputs=pdf_input,
        outputs=output_box
    )


if __name__ == "__main__":
    demo.launch()
