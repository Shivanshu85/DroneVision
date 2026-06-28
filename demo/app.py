from __future__ import annotations

import sys
from pathlib import Path

# Add the repository root directory to sys.path so we can import dronevision modules
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio as gr

from demo.config import (
    APPLICATION_DESC,
    APPLICATION_SUBTITLE,
    APPLICATION_TITLE,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    EXAMPLES_DIR,
)
from demo.predictor import DemoPredictor
from demo.ui import CSS, generate_metrics_html, generate_status_msg_html
from demo.utils import format_confidence_html

# Load predictor once at startup
predictor = DemoPredictor(checkpoint_path=DEFAULT_CHECKPOINT, config_path=DEFAULT_CONFIG)

def process_image(image):
    """
    Callback function when an image is uploaded or analyzed.
    
    Args:
        image: Numpy array of the input image or None.
        
    Returns:
        Updated Gradio components (Annotated Image, Metrics HTML, Table HTML, Status Msg, Error Msg).
    """
    if image is None:
        # Reset all outputs when image is cleared
        return (
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
        
    # Run prediction through the wrapper
    annotated_rgb, summary, error_msg = predictor.predict(image)
    
    if error_msg:
        # Render a user-friendly error message, hiding metrics and tables
        return (
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=generate_status_msg_html(error_msg, is_error=True), visible=True)
        )
        
    drone_count = summary["drone_count"]
    time_ms = summary["inference_time_ms"]
    device = summary["device"]
    confidences = summary["confidences"]
    
    # Generate HTML components
    metrics_html = generate_metrics_html(drone_count, time_ms, device)
    
    if drone_count > 0:
        table_html = format_confidence_html(confidences)
        return (
            annotated_rgb,
            gr.update(value=metrics_html, visible=True),
            gr.update(value=table_html, visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    else:
        # No detections case
        no_drones_html = generate_status_msg_html("No drones detected.")
        return (
            annotated_rgb,
            gr.update(value=metrics_html, visible=True),
            gr.update(visible=False),
            gr.update(value=no_drones_html, visible=True),
            gr.update(visible=False)
        )

# Construct example list from the demo/examples directory
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
examples_list = sorted([
    str(p) for p in EXAMPLES_DIR.iterdir()
    if p.suffix.lower() in image_extensions
])

# Create Gradio Blocks app with custom dark theme CSS
with gr.Blocks(title=f"{APPLICATION_TITLE} - Demo") as demo:
    # Styled wrapper container
    with gr.Column(elem_classes="main-container"):
        # Header block
        gr.HTML(
            f"""
            <div class="title-header">
                <p>{APPLICATION_DESC}</p>
                <h1>{APPLICATION_TITLE}</h1>
                <h2>{APPLICATION_SUBTITLE}</h2>
            </div>
            """
        )
        
        # Columns for input and output
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image", 
                    type="numpy",
                    sources=["upload"]
                )
                submit_btn = gr.Button("Detect Drones", variant="primary")
                
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Annotated Image", 
                    type="numpy",
                    interactive=False
                )
                
        # Status displays (error boxes, metrics, table)
        error_output = gr.HTML(visible=False)
        metrics_output = gr.HTML(visible=False)
        no_drones_output = gr.HTML(visible=False)
        table_output = gr.HTML(visible=False)
        
        # Click / upload interactions
        submit_btn.click(
            fn=process_image,
            inputs=input_image,
            outputs=[output_image, metrics_output, table_output, no_drones_output, error_output]
        )
        
        input_image.upload(
            fn=process_image,
            inputs=input_image,
            outputs=[output_image, metrics_output, table_output, no_drones_output, error_output]
        )
        
        input_image.clear(
            fn=lambda: (None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
            outputs=[output_image, metrics_output, table_output, no_drones_output, error_output]
        )
        
        # Examples section (rendered only if files exist)
        if examples_list:
            gr.Markdown("### Example Images")
            gr.Examples(
                examples=examples_list,
                inputs=input_image,
                outputs=[output_image, metrics_output, table_output, no_drones_output, error_output],
                fn=process_image,
                run_on_click=True
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=CSS)
