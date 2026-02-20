"""Image processing utilities for virtual try-on."""

import logging
from io import BytesIO
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Image processing utilities for try-on pipeline."""

    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image from file.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image object
        """
        try:
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Loaded image from {image_path}, size: {image.size}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    @staticmethod
    def resize_image(
        image: Image.Image, max_size: Tuple[int, int] = config.MAX_IMAGE_SIZE
    ) -> Image.Image:
        """
        Resize image to fit within max_size while maintaining aspect ratio.

        Args:
            image: PIL Image object
            max_size: Maximum (width, height)

        Returns:
            Resized PIL Image object
        """
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def normalize_image(image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to normalized numpy array.

        Args:
            image: PIL Image object

        Returns:
            Normalized numpy array (0-1 range, RGB)
        """
        image_array = np.array(image, dtype=np.float32) / 255.0
        return image_array

    @staticmethod
    def denormalize_image(
        image_array: np.ndarray, format: str = "PIL"
    ) -> Union[Image.Image, np.ndarray]:
        """
        Convert normalized numpy array back to image.

        Args:
            image_array: Normalized numpy array (0-1 range)
            format: Output format ('PIL' or 'numpy')

        Returns:
            PIL Image or numpy array (0-255)
        """
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)

        if format == "PIL":
            return Image.fromarray(image_array, mode="RGB")
        return image_array

    @staticmethod
    def crop_to_square(image: Image.Image) -> Image.Image:
        """
        Crop image to square (useful for consistent processing).

        Args:
            image: PIL Image object

        Returns:
            Squared PIL Image object
        """
        width, height = image.size
        size = min(width, height)
        left = (width - size) / 2
        top = (height - size) / 2
        right = left + size
        bottom = top + size
        return image.crop((left, top, right, bottom))

    @staticmethod
    def pad_to_square(image: Image.Image, fill_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
        """
        Pad image to square.

        Args:
            image: PIL Image object
            fill_color: RGB fill color

        Returns:
            Padded PIL Image object
        """
        width, height = image.size
        size = max(width, height)
        padded = Image.new("RGB", (size, size), fill_color)
        offset = ((size - width) // 2, (size - height) // 2)
        padded.paste(image, offset)
        return padded

    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded string
        """
        import base64

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()

    @staticmethod
    def base64_to_image(base64_string: str) -> Image.Image:
        """
        Convert base64 string to PIL Image.

        Args:
            base64_string: Base64 encoded image

        Returns:
            PIL Image object
        """
        import base64

        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data)).convert("RGB")

    @staticmethod
    def save_image(image: Image.Image, output_path: str, quality: int = config.IMAGE_QUALITY):
        """
        Save image to file.

        Args:
            image: PIL Image object
            output_path: Output file path
            quality: JPEG quality (1-100)
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, quality=quality)
            logger.info(f"Image saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise

    @staticmethod
    def apply_gaussian_blur(image: Image.Image, kernel_size: Tuple[int, int] = (5, 5)) -> Image.Image:
        """Apply Gaussian blur to image."""
        image_array = cv2.GaussianBlur(np.array(image), kernel_size, 0)
        return Image.fromarray(image_array)

    @staticmethod
    def enhance_contrast(image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Enhance image contrast."""
        from PIL import ImageEnhance

        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def enhance_brightness(image: Image.Image, factor: float = 1.1) -> Image.Image:
        """Enhance image brightness."""
        from PIL import ImageEnhance

        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)


# Global processor instance
_processor_instance = None


def get_processor() -> ImageProcessor:
    """Get or create the global processor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = ImageProcessor()
    return _processor_instance
