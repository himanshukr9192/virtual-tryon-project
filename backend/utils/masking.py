"""Semantic masking for intelligent garment placement."""

import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)


class SemanticMasker:
    """Generate semantic masks for garment placement."""

    @staticmethod
    def create_body_mask(image: Image.Image) -> np.ndarray:
        """
        Create a simple body mask using edge detection.
        This is a basic implementation; can be enhanced with ML models.

        Args:
            image: PIL Image object

        Returns:
            Binary mask (0 or 255) as numpy array
        """
        # Convert to grayscale
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Apply bilateral filter for smoothing while preserving edges
        filtered = cv2.bilateralFilter(image_array, 9, 75, 75)

        # Use edge detection
        edges = cv2.Canny(filtered, 100, 200)

        # Dilate edges to fill regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Fill contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(dilated)

        if contours:
            cv2.drawContours(mask, contours, -1, 255, -1)

        logger.debug("Body mask created successfully")
        return mask

    @staticmethod
    def create_garment_region_mask(
        image: Image.Image, garment_type: str, position: str = "center"
    ) -> np.ndarray:
        """
        Create a mask for specific garment placement regions.

        Args:
            image: PIL Image object
            garment_type: Type of clothing (shirt, hat, shoes, etc.)
            position: Garment position on body

        Returns:
            Binary mask for garment region
        """
        height, width = np.array(image).shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Define regions for different garment types
        garment_regions = {
            "hat": (0, 0, width, int(height * 0.15)),
            "glasses": (int(width * 0.25), int(height * 0.15), int(width * 0.75), int(height * 0.25)),
            "shirt": (0, int(height * 0.15), width, int(height * 0.6)),
            "t-shirt": (0, int(height * 0.15), width, int(height * 0.5)),
            "blouse": (0, int(height * 0.15), width, int(height * 0.55)),
            "dress": (0, int(height * 0.15), width, int(height * 0.85)),
            "jacket": (0, int(height * 0.15), width, int(height * 0.65)),
            "pants": (0, int(height * 0.45), width, int(height * 0.9)),
            "jeans": (0, int(height * 0.45), width, int(height * 0.9)),
            "skirt": (0, int(height * 0.4), width, int(height * 0.75)),
            "shoes": (0, int(height * 0.85), width, height),
            "scarf": (int(width * 0.2), int(height * 0.2), int(width * 0.8), int(height * 0.35)),
        }

        # Get region for garment type
        region = garment_regions.get(garment_type.lower(), (0, 0, width, height))
        x1, y1, x2, y2 = region

        # Draw mask region
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Apply dilation for smoothing
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.MASK_DILATION_KERNEL, config.MASK_DILATION_KERNEL)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Apply Gaussian blur for smooth edges
        mask = cv2.GaussianBlur(mask, config.MASK_BLUR_SIZE, 0)

        logger.debug(f"Garment region mask created for {garment_type}")
        return mask

    @staticmethod
    def refine_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
        """
        Refine mask using morphological operations.

        Args:
            mask: Binary mask
            iterations: Number of refinement iterations

        Returns:
            Refined mask
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Morphological closing (fill holes)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        # Morphological opening (remove noise)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

        return mask

    @staticmethod
    def blend_masks(mask1: np.ndarray, mask2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Blend two masks together.

        Args:
            mask1: First mask
            mask2: Second mask
            alpha: Blending factor

        Returns:
            Blended mask
        """
        blended = cv2.addWeighted(mask1, alpha, mask2, 1 - alpha, 0).astype(np.uint8)
        return blended

    @staticmethod
    def apply_mask_to_image(
        image: Image.Image, mask: np.ndarray, blend_alpha: float = 0.7
    ) -> Image.Image:
        """
        Apply mask to image for visualization.

        Args:
            image: PIL Image object
            mask: Binary mask
            blend_alpha: Alpha blending factor

        Returns:
            Masked image as PIL Image
        """
        image_array = np.array(image)
        mask_normalized = mask / 255.0

        # Create a purple overlay for masked region
        overlay = image_array.copy()
        overlay[mask > 127] = [200, 100, 200]  # Purple color

        # Blend
        result = (image_array * (1 - blend_alpha) + overlay * blend_alpha).astype(np.uint8)

        return Image.fromarray(result)

    @staticmethod
    def get_mask_statistics(mask: np.ndarray) -> dict:
        """Get statistics about a mask."""
        total_pixels = mask.size
        masked_pixels = np.count_nonzero(mask)
        coverage = (masked_pixels / total_pixels) * 100

        return {
            "total_pixels": total_pixels,
            "masked_pixels": masked_pixels,
            "coverage_percentage": coverage,
        }


# Global masker instance
_masker_instance = None


def get_masker() -> SemanticMasker:
    """Get or create the global masker instance."""
    global _masker_instance
    if _masker_instance is None:
        _masker_instance = SemanticMasker()
    return _masker_instance
