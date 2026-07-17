"""
detector.py

The core object detection engine of the project.

This module defines the ObjectDetector class, which ties together the
model (loaded via model_loader.py) and the image utilities (from
image_utils.py) to run the full Input -> Process -> Output detection
pipeline:

    Input:   an image path + a confidence threshold
    Process: load image -> run MobileNet-SSD inference -> filter by
             confidence -> build detection results -> draw boxes
    Output:  an annotated image + a structured detection summary
"""

import cv2
import numpy as np

import config
import image_utils
from logger_setup import get_logger
from model_loader import ModelLoader

logger = get_logger(__name__)


class ObjectDetector:
    """
    Runs object detection on images using a pre-trained MobileNet-SSD
    model.

    This class is the "engine" of the project: it does not concern
    itself with user interaction (that's main.py's job) or with the
    low-level details of loading model files (that's model_loader.py's
    job). Its single responsibility is: given a loaded model, an image,
    and a confidence threshold, produce detection results.
    """

    def __init__(self, confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        """
        Args:
            confidence_threshold: Minimum confidence (0.0-1.0) a
                detection must have to be kept. Defaults to the value
                defined in config.py.
        """
        self.confidence_threshold: float = confidence_threshold
        self._model_loader: ModelLoader = ModelLoader()
        self.network: cv2.dnn_Net | None = None

    def load_model(self) -> None:
        """
        Load the MobileNet-SSD model into memory via ModelLoader.

        Must be called once before running any detections. Kept as a
        separate step (rather than doing it in __init__) so that model
        loading failures can be handled explicitly by the caller
        (main.py), with a clear try/except around this specific step.
        """
        self.network = self._model_loader.load_model()

    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update the confidence threshold used to filter detections.

        Args:
            threshold: New threshold value, must be between
                config.MIN_CONFIDENCE_THRESHOLD and
                config.MAX_CONFIDENCE_THRESHOLD.

        Raises:
            ValueError: If the threshold is outside the valid range.
        """
        if not (config.MIN_CONFIDENCE_THRESHOLD <= threshold <= config.MAX_CONFIDENCE_THRESHOLD):
            raise ValueError(
                f"Confidence threshold must be between "
                f"{config.MIN_CONFIDENCE_THRESHOLD} and {config.MAX_CONFIDENCE_THRESHOLD}."
            )
        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold updated to {threshold:.2f}")

    def _run_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Run a forward pass of the MobileNet-SSD network on the image.

        Args:
            image: Input image (BGR NumPy array).

        Returns:
            The raw output array from the network, containing one row
            per candidate detection with class ID, confidence, and box
            coordinates (normalized 0-1).
        """
        blob = cv2.dnn.blobFromImage(
            image,
            config.SCALE_FACTOR,
            config.INPUT_IMAGE_SIZE,
            config.MEAN_SUBTRACTION,
        )
        self.network.setInput(blob)
        raw_output = self.network.forward()
        return raw_output

    def _filter_and_format_detections(
        self, raw_output: np.ndarray, image_width: int, image_height: int
    ) -> list[dict]:
        """
        Filter raw network output by confidence threshold and convert
        it into a clean list of detection dictionaries with pixel-space
        bounding boxes.

        Args:
            raw_output: The raw output array from _run_inference().
            image_width: Width of the original image, used to scale
                normalized box coordinates back to pixel coordinates.
            image_height: Height of the original image.

        Returns:
            A list of dicts, each with keys: 'label', 'confidence',
            and 'box' (x1, y1, x2, y2 in pixel coordinates).
        """
        detections: list[dict] = []

        # raw_output shape is [1, 1, num_detections, 7]; each row is:
        # [batch_id, class_id, confidence, x1, y1, x2, y2] (normalized).
        for i in range(raw_output.shape[2]):
            confidence = float(raw_output[0, 0, i, 2])

            if confidence < self.confidence_threshold:
                continue

            class_id = int(raw_output[0, 0, i, 1])
            if class_id < 0 or class_id >= len(config.CLASS_LABELS):
                # Skip any class ID outside the known label range.
                continue

            label = config.CLASS_LABELS[class_id]

            # Convert normalized coordinates (0-1) to pixel coordinates.
            box = raw_output[0, 0, i, 3:7] * np.array(
                [image_width, image_height, image_width, image_height]
            )
            x1, y1, x2, y2 = box.astype(int)

            # Clip coordinates so boxes never fall outside the image.
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_width - 1, x2), min(image_height - 1, y2)

            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                }
            )

        return detections

    def detect(self, image_path: str) -> dict:
        """
        Run the full detection pipeline on a single image.

        Args:
            image_path: Path to the input image.

        Returns:
            A dictionary with keys:
                - 'original_image_path': the input path
                - 'annotated_image': the image with boxes drawn (NumPy array)
                - 'detections': list of detection dicts (label, confidence, box)
                - 'object_count': total number of objects detected

        Raises:
            RuntimeError: If detect() is called before load_model().
            image_utils.InvalidImageError: If the image is invalid.
        """
        if self.network is None:
            raise RuntimeError(
                "Model is not loaded. Call load_model() before detect()."
            )

        image = image_utils.load_image(image_path)
        image_height, image_width = image.shape[:2]

        raw_output = self._run_inference(image)
        detections = self._filter_and_format_detections(raw_output, image_width, image_height)

        annotated_image = image_utils.draw_detections(image, detections)

        logger.info(
            f"Detection complete on '{image_path}': {len(detections)} object(s) "
            f"found above {self.confidence_threshold:.2f} confidence threshold."
        )

        return {
            "original_image_path": image_path,
            "annotated_image": annotated_image,
            "detections": detections,
            "object_count": len(detections),
        }
