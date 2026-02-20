"""Unit tests for Virtual Try-On AI."""

import unittest
from PIL import Image
import numpy as np

from backend.models import ClothingClassifier
from backend.utils.image_processor import ImageProcessor
from backend.utils.masking import SemanticMasker


class TestImageProcessor(unittest.TestCase):
    """Tests for ImageProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor()
        # Create a dummy image
        self.test_image = Image.new("RGB", (512, 512), color="red")

    def test_normalize_image(self):
        """Test image normalization."""
        normalized = self.processor.normalize_image(self.test_image)
        self.assertEqual(normalized.shape, (512, 512, 3))
        self.assertTrue(np.all(normalized <= 1.0))
        self.assertTrue(np.all(normalized >= 0.0))

    def test_denormalize_image(self):
        """Test image denormalization."""
        normalized = self.processor.normalize_image(self.test_image)
        denormalized = self.processor.denormalize_image(normalized, format="PIL")
        self.assertIsInstance(denormalized, Image.Image)

    def test_image_to_base64_and_back(self):
        """Test base64 conversion."""
        base64_str = self.processor.image_to_base64(self.test_image)
        recovered_image = self.processor.base64_to_image(base64_str)
        self.assertEqual(recovered_image.size, self.test_image.size)


class TestSemanticMasker(unittest.TestCase):
    """Tests for SemanticMasker."""

    def setUp(self):
        """Set up test fixtures."""
        self.masker = SemanticMasker()
        self.test_image = Image.new("RGB", (512, 512), color="blue")

    def test_create_garment_region_mask(self):
        """Test mask creation for different garment types."""
        garment_types = ["hat", "shirt", "pants", "shoes"]

        for garment_type in garment_types:
            mask = self.masker.create_garment_region_mask(
                self.test_image, garment_type
            )
            self.assertEqual(mask.shape, (512, 512))
            self.assertTrue(np.max(mask) > 0)  # Has masked region

    def test_refine_mask(self):
        """Test mask refinement."""
        mask = self.masker.create_garment_region_mask(self.test_image, "shirt")
        refined = self.masker.refine_mask(mask)
        self.assertEqual(refined.shape, mask.shape)

    def test_get_mask_statistics(self):
        """Test mask statistics calculation."""
        mask = self.masker.create_garment_region_mask(self.test_image, "shirt")
        stats = self.masker.get_mask_statistics(mask)

        self.assertIn("coverage_percentage", stats)
        self.assertIn("masked_pixels", stats)
        self.assertIn("total_pixels", stats)
        self.assertGreater(stats["coverage_percentage"], 0)


class TestClothingClassifier(unittest.TestCase):
    """Tests for ClothingClassifier."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        try:
            cls.classifier = ClothingClassifier()
            cls.skip_tests = False
        except Exception as e:
            print(f"Could not load CLIP model: {e}")
            cls.skip_tests = True

    def setUp(self):
        """Set up test fixtures."""
        if not self.skip_tests:
            self.test_image = Image.new("RGB", (224, 224), color="green")

    def test_classify(self):
        """Test image classification."""
        if self.skip_tests:
            self.skipTest("CLIP model not available")

        result = self.classifier.classify(self.test_image)

        self.assertIn("garment_type", result)
        self.assertIn("confidence", result)
        self.assertIn("alternatives", result)
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 1)


if __name__ == "__main__":
    unittest.main()
