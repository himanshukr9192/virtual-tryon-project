"""Inference pipeline for virtual try-on."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

from backend.models import ClothingClassifier
from backend.utils.image_processor import ImageProcessor
from backend.utils.masking import SemanticMasker
import config

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Complete inference pipeline for virtual try-on."""

    def __init__(self, device: str = config.DEVICE):
        """
        Initialize the inference pipeline.

        Args:
            device: Computation device (cuda/cpu)
        """
        self.device = device
        self.classifier = ClothingClassifier(device=device)
        self.processor = ImageProcessor()
        self.masker = SemanticMasker()

        # Initialize Stable Diffusion inpainting pipeline
        try:
            logger.info(f"Loading Stable Diffusion inpainting model on {device}")
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                config.STABLE_DIFFUSION_MODEL_ID,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                cache_dir=config.MODEL_CACHE_DIR,
            ).to(device)
            self.pipeline.enable_attention_slicing()  # For memory efficiency
            logger.info("Stable Diffusion pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion pipeline: {e}")
            raise

    def classify_garment(self, garment_image: Image.Image) -> Dict:
        """
        Classify the garment type.

        Args:
            garment_image: PIL Image of the garment

        Returns:
            Classification results
        """
        return self.classifier.classify(garment_image)

    def generate_try_on(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_type: Optional[str] = None,
        guided_scale: float = config.GUIDANCE_SCALE,
        num_inference_steps: int = config.NUM_INFERENCE_STEPS,
    ) -> Dict:
        """
        Generate virtual try-on result.

        Args:
            person_image: PIL Image of the person
            garment_image: PIL Image of the garment
            garment_type: Type of garment (auto-detected if None)
            guided_scale: Guidance scale for diffusion
            num_inference_steps: Number of inference steps

        Returns:
            Dictionary with:
                - result_image: Generated try-on image
                - garment_detected: Detected garment type
                - confidence: Detection confidence
                - mask: Used mask
                - processing_time: Time taken for generation
        """
        import time

        start_time = time.time()

        try:
            # Resize images
            person_image = self.processor.resize_image(person_image)
            garment_image = self.processor.resize_image(garment_image)

            # Auto-classify if not provided
            if garment_type is None:
                classification = self.classify_garment(garment_image)
                garment_type = classification["garment_type"]
                confidence = classification["confidence"]
            else:
                confidence = 1.0

            logger.info(f"Processing try-on for {garment_type} (confidence: {confidence:.2%})")

            # Create semantic mask
            mask = self.masker.create_garment_region_mask(person_image, garment_type)
            mask = self.masker.refine_mask(mask)

            # Convert to PIL Image for inpainting
            mask_image = Image.fromarray(mask).convert("L")

            # Create prompt based on garment type
            prompt = self._generate_prompt(garment_type, "person wearing high-quality clothing")

            logger.info(f"Inference prompt: {prompt}")

            # Run inpainting
            with torch.no_grad():
                result_image = self.pipeline(
                    prompt=prompt,
                    image=person_image,
                    mask_image=mask_image,
                    guidance_scale=guided_scale,
                    num_inference_steps=num_inference_steps,
                ).images[0]

            processing_time = time.time() - start_time

            result = {
                "result_image": result_image,
                "garment_detected": garment_type,
                "confidence": confidence,
                "mask": mask_image,
                "processing_time": processing_time,
            }

            logger.info(f"Try-on generation completed in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error during try-on generation: {e}")
            raise

    def _generate_prompt(self, garment_type: str, base_prompt: str) -> str:
        """
        Generate a detailed prompt for Stable Diffusion.

        Args:
            garment_type: Type of garment
            base_prompt: Base prompt text

        Returns:
            Generated prompt
        """
        garment_prompts = {
            "hat": "wearing a stylish hat",
            "cap": "wearing a cap",
            "glasses": "wearing glasses",
            "sunglasses": "wearing sunglasses",
            "shirt": "wearing a button-up shirt",
            "t-shirt": "wearing a t-shirt",
            "blouse": "wearing a blouse",
            "dress": "wearing a dress",
            "jacket": "wearing a jacket",
            "coat": "wearing a coat",
            "sweater": "wearing a sweater",
            "hoodie": "wearing a hoodie",
            "pants": "wearing pants",
            "jeans": "wearing jeans",
            "skirt": "wearing a skirt",
            "shoes": "wearing shoes",
            "scarf": "wearing a scarf",
            "tie": "wearing a tie",
        }

        garment_detail = garment_prompts.get(garment_type.lower(), f"wearing {garment_type}")

        prompt = (
            f"A person {garment_detail}, photorealistic, high quality, "
            f"professional photography, natural lighting, detailed fabric texture"
        )

        return prompt

    def batch_try_on(
        self,
        person_images: list,
        garment_images: list,
        garment_types: list = None,
    ) -> list:
        """
        Process multiple try-on requests.

        Args:
            person_images: List of person images
            garment_images: List of garment images
            garment_types: List of garment types (optional)

        Returns:
            List of results
        """
        results = []
        for i, (person_img, garment_img) in enumerate(zip(person_images, garment_images)):
            garment_type = garment_types[i] if garment_types else None
            try:
                result = self.generate_try_on(person_img, garment_img, garment_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing batch item {i}: {e}")
                results.append({"error": str(e)})

        return results

    def clear_cache(self):
        """Clear GPU/CPU cache to free memory."""
        torch.cuda.empty_cache() if self.device == "cuda" else None
        logger.info("Cache cleared")


# Global pipeline instance
_pipeline_instance = None


def get_pipeline() -> InferencePipeline:
    """Get or create the global pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = InferencePipeline()
    return _pipeline_instance
