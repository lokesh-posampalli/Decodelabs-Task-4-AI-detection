"""
tests/test_detector.py

Unit tests for the AI Object Detection System, using Python's built-in
unittest framework.

These tests focus on the parts of the system that can be tested
reliably without requiring a live webcam or a large batch of real
photos: input validation, threshold filtering logic, history
management, and (where model files are available) basic model loading
and end-to-end detection.

Run with:
    python -m unittest tests/test_detector.py -v

Note: Tests that require the actual MobileNet-SSD model files will be
automatically skipped if those files are not present in models/, so
this suite can still run (and mostly pass) on a fresh clone of the
project before the model has been downloaded.
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

# Allow imports from the project root when running tests directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import image_utils
from detection_history import DetectionHistory
from detector import ObjectDetector
from model_loader import ModelLoader, ModelLoadError


MODEL_FILES_AVAILABLE = os.path.isfile(config.PROTOTXT_PATH) and os.path.isfile(
    config.MODEL_WEIGHTS_PATH
)


class TestModelLoading(unittest.TestCase):
    """Tests for model_loader.ModelLoader."""

    def test_verify_missing_model_files_raises_error(self) -> None:
        """Verification should fail clearly when model files don't exist."""
        loader = ModelLoader(
            prototxt_path="models/does_not_exist.prototxt",
            weights_path="models/does_not_exist.caffemodel",
        )
        with self.assertRaises(ModelLoadError):
            loader.verify_model_files()

    @unittest.skipUnless(MODEL_FILES_AVAILABLE, "MobileNet-SSD model files not present.")
    def test_load_real_model_succeeds(self) -> None:
        """When model files ARE present, loading should succeed."""
        loader = ModelLoader()
        network = loader.load_model()
        self.assertIsNotNone(network)
        self.assertTrue(loader.is_loaded())


class TestImageValidation(unittest.TestCase):
    """Tests for image_utils validation and I/O functions."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_missing_file_raises_error(self) -> None:
        with self.assertRaises(image_utils.InvalidImageError):
            image_utils.validate_image_path(
                os.path.join(self.temp_dir, "nonexistent.jpg")
            )

    def test_validate_unsupported_extension_raises_error(self) -> None:
        bad_file_path = os.path.join(self.temp_dir, "document.txt")
        with open(bad_file_path, "w", encoding="utf-8") as file:
            file.write("not an image")

        with self.assertRaises(image_utils.InvalidImageError):
            image_utils.validate_image_path(bad_file_path)

    def test_validate_case_insensitive_extension(self) -> None:
        """A .JPG (uppercase) extension should be treated as valid."""
        self.assertTrue(image_utils.is_supported_extension("photo.JPG"))
        self.assertTrue(image_utils.is_supported_extension("photo.PnG"))

    def test_validate_empty_path_raises_error(self) -> None:
        with self.assertRaises(image_utils.InvalidImageError):
            image_utils.validate_image_path("   ")


class TestBoundingBoxDrawing(unittest.TestCase):
    """Tests for image_utils.draw_detections()."""

    def setUp(self) -> None:
        # A small blank black image to draw on.
        self.blank_image = np.zeros((200, 200, 3), dtype=np.uint8)

    def test_draw_detections_returns_modified_copy(self) -> None:
        detections = [
            {"label": "person", "confidence": 0.95, "box": (10, 10, 100, 150)}
        ]
        annotated = image_utils.draw_detections(self.blank_image, detections)

        # The original image must remain untouched (still all zeros).
        self.assertTrue(np.array_equal(self.blank_image, np.zeros((200, 200, 3), dtype=np.uint8)))
        # The annotated image must differ from the original (a box was drawn).
        self.assertFalse(np.array_equal(annotated, self.blank_image))

    def test_draw_detections_with_no_objects_returns_unchanged_copy(self) -> None:
        annotated = image_utils.draw_detections(self.blank_image, [])
        self.assertTrue(np.array_equal(annotated, self.blank_image))


class TestConfidenceThresholdFiltering(unittest.TestCase):
    """Tests for ObjectDetector's threshold validation and filtering logic."""

    def setUp(self) -> None:
        self.detector = ObjectDetector()

    def test_default_threshold_matches_config(self) -> None:
        self.assertEqual(self.detector.confidence_threshold, config.DEFAULT_CONFIDENCE_THRESHOLD)

    def test_set_valid_threshold_updates_value(self) -> None:
        self.detector.set_confidence_threshold(0.75)
        self.assertEqual(self.detector.confidence_threshold, 0.75)

    def test_set_threshold_below_range_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            self.detector.set_confidence_threshold(-0.1)

    def test_set_threshold_above_range_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            self.detector.set_confidence_threshold(1.1)

    def test_filter_detections_respects_threshold(self) -> None:
        """
        Build a fake raw network output manually to test the private
        filtering method without needing the real model.
        """
        self.detector.confidence_threshold = 0.60

        # Shape [1, 1, num_detections, 7]: batch_id, class_id, confidence, x1, y1, x2, y2
        raw_output = np.array(
            [[[
                [0, 15, 0.90, 0.1, 0.1, 0.5, 0.5],   # person, above threshold
                [0, 7, 0.40, 0.2, 0.2, 0.6, 0.6],    # car, below threshold
            ]]],
            dtype=np.float32,
        )

        results = self.detector._filter_and_format_detections(
            raw_output, image_width=200, image_height=200
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["label"], "person")
        self.assertGreaterEqual(results[0]["confidence"], 0.60)

    def test_detect_before_load_model_raises_runtime_error(self) -> None:
        with self.assertRaises(RuntimeError):
            self.detector.detect("images/sample.jpg")


class TestDetectionHistory(unittest.TestCase):
    """Tests for detection_history.DetectionHistory."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.history_path = os.path.join(self.temp_dir, "history.json")
        self.history = DetectionHistory(history_file_path=self.history_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_new_history_starts_empty(self) -> None:
        self.assertEqual(self.history.total_entries(), 0)

    def test_add_entry_increases_count(self) -> None:
        detections = [{"label": "dog", "confidence": 0.88, "box": (0, 0, 10, 10)}]
        self.history.add_entry("images/dog.jpg", detections)
        self.assertEqual(self.history.total_entries(), 1)

    def test_add_entry_persists_to_disk(self) -> None:
        detections = [{"label": "cat", "confidence": 0.77, "box": (0, 0, 10, 10)}]
        self.history.add_entry("images/cat.jpg", detections)

        reloaded_history = DetectionHistory(history_file_path=self.history_path)
        self.assertEqual(reloaded_history.total_entries(), 1)
        self.assertEqual(reloaded_history.get_all_entries()[0]["image_name"], "cat.jpg")

    def test_clear_history_removes_all_entries(self) -> None:
        self.history.add_entry("images/bird.jpg", [])
        self.history.clear_history()
        self.assertEqual(self.history.total_entries(), 0)

    def test_history_cap_trims_oldest_entries(self) -> None:
        original_cap = config.MAX_HISTORY_ENTRIES
        try:
            config.MAX_HISTORY_ENTRIES = 3
            for i in range(5):
                self.history.add_entry(f"images/image_{i}.jpg", [])

            self.assertEqual(self.history.total_entries(), 3)
            # The three most recent entries (2, 3, 4) should be kept.
            remaining_names = [entry["image_name"] for entry in self.history.get_all_entries()]
            self.assertEqual(remaining_names, ["image_2.jpg", "image_3.jpg", "image_4.jpg"])
        finally:
            config.MAX_HISTORY_ENTRIES = original_cap


class TestEndToEndDetection(unittest.TestCase):
    """
    End-to-end test of the full detection pipeline. Only runs if the
    real MobileNet-SSD model files are present in models/.
    """

    @unittest.skipUnless(MODEL_FILES_AVAILABLE, "MobileNet-SSD model files not present.")
    def test_detect_on_blank_image_runs_without_error(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            import cv2

            image_path = os.path.join(temp_dir, "blank.jpg")
            blank_image = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.imwrite(image_path, blank_image)

            detector = ObjectDetector()
            detector.load_model()
            result = detector.detect(image_path)

            self.assertIn("object_count", result)
            self.assertIn("annotated_image", result)
            self.assertIsInstance(result["detections"], list)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
