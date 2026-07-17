"""
image_utils.py

Utility functions for handling images: validating file paths, loading
images from disk, drawing bounding boxes with labels, and saving the
final annotated output.

These are kept separate from detector.py so that "image plumbing"
(reading/writing/drawing) stays isolated from the actual AI inference
logic. This separation makes both files easier to read, test, and
reuse independently.
"""

import os
from datetime import datetime

import cv2
import numpy as np

import config
from logger_setup import get_logger

logger = get_logger(__name__)


class InvalidImageError(Exception):
    """
    Raised when an image path does not exist, has an unsupported
    extension, or cannot be read/decoded by OpenCV.
    """
    pass


def is_supported_extension(image_path: str) -> bool:
    """
    Check whether a file has a supported image extension.

    The check is case-insensitive, so 'photo.JPG' and 'photo.jpg' are
    both treated as valid.

    Args:
        image_path: Path to the image file.

    Returns:
        True if the extension is supported, False otherwise.
    """
    extension = os.path.splitext(image_path)[1].lower()
    return extension in config.SUPPORTED_IMAGE_EXTENSIONS


def validate_image_path(image_path: str) -> str:
    """
    Validate that an image path exists and has a supported extension.

    Args:
        image_path: Path to the image file, as entered by the user.

    Returns:
        The cleaned (whitespace-stripped) image path if valid.

    Raises:
        InvalidImageError: If the path is empty, doesn't exist, or has
            an unsupported file extension.
    """
    cleaned_path = image_path.strip().strip('"').strip("'")

    if not cleaned_path:
        raise InvalidImageError("Image path cannot be empty.")

    if not os.path.isfile(cleaned_path):
        raise InvalidImageError(f"Image file not found: {cleaned_path}")

    if not is_supported_extension(cleaned_path):
        supported = ", ".join(config.SUPPORTED_IMAGE_EXTENSIONS)
        raise InvalidImageError(
            f"Unsupported image format: '{cleaned_path}'. "
            f"Supported formats are: {supported}"
        )

    return cleaned_path


def load_image(image_path: str) -> np.ndarray:
    """
    Validate and load an image from disk into a NumPy array (BGR format,
    as used by OpenCV).

    Args:
        image_path: Path to the image file.

    Returns:
        The loaded image as a NumPy array.

    Raises:
        InvalidImageError: If the path is invalid or OpenCV fails to
            decode the file (e.g. it's corrupted or not really an image).
    """
    validated_path = validate_image_path(image_path)

    image = cv2.imread(validated_path)
    if image is None:
        raise InvalidImageError(
            f"Failed to read image (file may be corrupted): {validated_path}"
        )

    logger.info(f"Image loaded successfully: {validated_path}")
    return image


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes, labels, and confidence percentages for a list
    of detected objects onto a copy of the given image.

    Args:
        image: The original image (BGR NumPy array).
        detections: A list of detection dictionaries, each expected to
            have the keys: 'label' (str), 'confidence' (float, 0-1),
            'box' (tuple of x1, y1, x2, y2).

    Returns:
        A new image (copy of the input) with bounding boxes and labels
        drawn on it. The original image is left untouched.
    """
    annotated_image = image.copy()

    for detection in detections:
        label = detection["label"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = detection["box"]

        # Draw the bounding box rectangle.
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            config.BOUNDING_BOX_COLOR,
            config.BOX_THICKNESS,
        )

        # Build a readable label string, e.g. "Person: 98.43%".
        label_text = f"{label}: {confidence * 100:.2f}%"

        # Place the label above the box, unless the box is too close
        # to the top edge, in which case place it just below instead.
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20

        # Draw a filled rectangle behind the text so it stays readable
        # over busy image backgrounds.
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
        )
        cv2.rectangle(
            annotated_image,
            (x1, text_y - text_height - 4),
            (x1 + text_width + 4, text_y + 4),
            config.BOUNDING_BOX_COLOR,
            cv2.FILLED,
        )
        cv2.putText(
            annotated_image,
            label_text,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            config.TEXT_COLOR,
            config.FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


def save_output_image(image: np.ndarray, original_image_path: str) -> str:
    """
    Save an annotated image to the output/ folder with a timestamped,
    traceable filename.

    Args:
        image: The annotated image to save.
        original_image_path: The path of the original input image, used
            to build a meaningful output filename.

    Returns:
        The full path where the output image was saved.

    Raises:
        InvalidImageError: If OpenCV fails to write the file to disk.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    original_name = os.path.splitext(os.path.basename(original_image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{original_name}_detected_{timestamp}.jpg"
    output_path = os.path.join(config.OUTPUT_DIR, output_filename)

    success = cv2.imwrite(output_path, image)
    if not success:
        raise InvalidImageError(f"Failed to save output image to: {output_path}")

    logger.info(f"Annotated image saved to: {output_path}")
    return output_path
