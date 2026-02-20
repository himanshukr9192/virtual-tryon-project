"""Gradio interface for Virtual Try-On AI."""

import logging
from io import BytesIO
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from backend.utils.inference import get_pipeline
from backend.utils.image_processor import ImageProcessor

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize components
pipeline = get_pipeline()
processor = ImageProcessor()


def process_try_on(
    person_image: np.ndarray,
    garment_image: np.ndarray,
    garment_type: str,
) -> tuple:
    """
    Process virtual try-on request.

    Args:
        person_image: Person image as numpy array
        garment_image: Garment image as numpy array
        garment_type: Selected or custom garment type

    Returns:
        Tuple of (result_image, status_text)
    """
    try:
        if person_image is None or garment_image is None:
            return None, "❌ Please upload both person and garment images"

        # Convert to PIL Images
        person_pil = Image.fromarray(person_image.astype("uint8"), "RGB")
        garment_pil = Image.fromarray(garment_image.astype("uint8"), "RGB")

        # Process
        logger.info(f"Processing try-on with garment type: {garment_type}")
        result = pipeline.generate_try_on(
            person_image=person_pil,
            garment_image=garment_pil,
            garment_type=garment_type if garment_type != "Auto-detect" else None,
        )

        status = (
            f"✅ Try-on completed!\n\n"
            f"**Garment detected:** {result['garment_detected']}\n"
            f"**Confidence:** {result['confidence']:.1%}\n"
            f"**Processing time:** {result['processing_time']:.2f}s"
        )

        return result["result_image"], status

    except Exception as e:
        logger.error(f"Error processing try-on: {e}")
        return None, f"❌ Error: {str(e)}"


def classify_garment_ui(garment_image: np.ndarray) -> str:
    """
    Classify garment type from image.

    Args:
        garment_image: Garment image as numpy array

    Returns:
        Classification result text
    """
    try:
        if garment_image is None:
            return "❌ Please upload a garment image"

        garment_pil = Image.fromarray(garment_image.astype("uint8"), "RGB")
        result = pipeline.classify_garment(garment_pil)

        classification_text = (
            f"**Detected Garment:** {result['garment_type']}\n"
            f"**Confidence:** {result['confidence']:.1%}\n\n"
            f"**Other possibilities:**\n"
        )

        for i, alt in enumerate(result["alternatives"], 1):
            classification_text += f"{i}. {alt}\n"

        return classification_text

    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return f"❌ Error: {str(e)}"


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(title="Virtual Try-On AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 👗 Virtual Try-On AI
        
        AI-powered virtual clothing try-on using advanced machine learning models.
        Upload a photo of yourself and a clothing item to see how it would look on you!
        """)

        with gr.Tabs():
            # Try-On Tab
            with gr.Tab("Try-On"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Person Image")
                        person_input = gr.Image(
                            label="Upload your photo",
                            type="numpy",
                            sources=["upload", "webcam"],
                        )

                    with gr.Column():
                        gr.Markdown("### Garment Image")
                        garment_input = gr.Image(
                            label="Upload garment photo",
                            type="numpy",
                            sources=["upload"],
                        )

                with gr.Row():
                    with gr.Column(scale=2):
                        garment_type_input = gr.Dropdown(
                            choices=["Auto-detect"] + config.GARMENT_TYPES,
                            value="Auto-detect",
                            label="Garment Type",
                            info="Select or auto-detect the garment type",
                        )

                    with gr.Column(scale=1):
                        tryon_button = gr.Button(
                            "🎨 Generate Try-On",
                            variant="primary",
                            size="lg",
                        )

                with gr.Row():
                    with gr.Column():
                        result_image = gr.Image(
                            label="Try-On Result",
                            type="pil",
                        )

                    with gr.Column():
                        status_output = gr.Markdown("Waiting for input...")

                tryon_button.click(
                    fn=process_try_on,
                    inputs=[person_input, garment_input, garment_type_input],
                    outputs=[result_image, status_output],
                )

            # Classification Tab
            with gr.Tab("Garment Classification"):
                gr.Markdown("""
                ### Identify Clothing Types
                Upload a clothing item image to automatically classify it.
                """)

                with gr.Row():
                    with gr.Column():
                        classify_input = gr.Image(
                            label="Upload garment image",
                            type="numpy",
                            sources=["upload"],
                        )

                    with gr.Column():
                        classify_button = gr.Button(
                            "🔍 Classify",
                            variant="primary",
                            size="lg",
                        )

                classification_output = gr.Markdown("Upload an image to classify...")

                classify_button.click(
                    fn=classify_garment_ui,
                    inputs=[classify_input],
                    outputs=[classification_output],
                )

            # About Tab
            with gr.Tab("About"):
                gr.Markdown("""
                ## About Virtual Try-On AI
                
                This application uses state-of-the-art AI models to enable virtual clothing try-ons:
                
                ### Key Features
                - **Automatic Clothing Detection**: Uses CLIP for zero-shot classification
                - **Semantic Masking**: Intelligently identifies where to place garments
                - **Realistic Rendering**: Powered by Stable Diffusion inpainting
                - **User-Friendly Interface**: Simple and intuitive Gradio UI
                
                ### Technology Stack
                - **ML Framework**: PyTorch & HuggingFace Transformers
                - **Models**:
                  - OpenAI CLIP for classification
                  - Stable Diffusion for image generation
                - **Frontend**: Gradio
                - **Backend**: FastAPI
                
                ### Supported Garment Types
                """ + ", ".join(config.GARMENT_TYPES) + """
                
                ---
                
                **Disclaimer**: This is a research/demo project. Results may vary based on image
                quality, lighting, and other factors.
                """)

    return demo


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name=config.GRADIO_HOST,
        server_port=config.GRADIO_PORT,
        share=config.GRADIO_SHARE,
    )
