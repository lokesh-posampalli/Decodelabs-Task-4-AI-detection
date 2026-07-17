"""
model_loader.py

Responsible for locating, verifying, and loading the pre-trained
MobileNet-SSD model into OpenCV's DNN module.

This module keeps all "model bookkeeping" in one place: checking that
the required files exist, giving clear error messages if they don't,
and returning a ready-to-use OpenCV network object. Separating this
from the detection logic (detector.py) keeps each file focused on a
single responsibility.
"""

import os

import cv2

import config
from logger_setup import get_logger

logger = get_logger(__name__)


class ModelLoadError(Exception):
    """
    Raised when the MobileNet-SSD model files are missing, unreadable,
    or fail to load into OpenCV's DNN module.

    Using a custom exception (instead of letting a generic OSError or
    cv2.error bubble up) makes it clear to calling code exactly what
    went wrong, and lets main.py catch this specific failure mode.
    """
    pass


class ModelLoader:
    """
    Handles verification and loading of the pre-trained MobileNet-SSD
    model files (the .prototxt architecture file and the .caffemodel
    weights file).

    This class does NOT perform any object detection itself — its only
    responsibility is making sure the model is present on disk and
    successfully loaded into memory via OpenCV's DNN module.
    """

    def __init__(
        self,
        prototxt_path: str = config.PROTOTXT_PATH,
        weights_path: str = config.MODEL_WEIGHTS_PATH,
    ) -> None:
        """
        Args:
            prototxt_path: Path to the MobileNet-SSD .prototxt file.
            weights_path: Path to the MobileNet-SSD .caffemodel file.
        """
        self.prototxt_path: str = prototxt_path
        self.weights_path: str = weights_path
        self.network: cv2.dnn_Net | None = None

    def verify_model_files(self) -> bool:
        """
        Check that both required model files exist and are non-empty.

        Returns:
            True if both files exist and appear valid.

        Raises:
            ModelLoadError: If either file is missing or empty, with a
                message explaining exactly which file is the problem.
        """
        missing_files: list[str] = []

        if not os.path.isfile(self.prototxt_path):
            missing_files.append(self.prototxt_path)
        elif os.path.getsize(self.prototxt_path) == 0:
            raise ModelLoadError(
                f"Model architecture file is empty: {self.prototxt_path}"
            )

        if not os.path.isfile(self.weights_path):
            missing_files.append(self.weights_path)
        elif os.path.getsize(self.weights_path) == 0:
            raise ModelLoadError(
                f"Model weights file is empty: {self.weights_path}"
            )

        if missing_files:
            files_list = "\n  - ".join(missing_files)
            raise ModelLoadError(
                "Required MobileNet-SSD model file(s) not found:\n  - "
                f"{files_list}\n"
                "Please download the pre-trained MobileNet-SSD files and "
                "place them inside the 'models/' folder. See README.md "
                "for download instructions."
            )

        logger.info("Model files verified successfully.")
        return True

    def load_model(self) -> cv2.dnn_Net:
        """
        Verify and load the MobileNet-SSD model into OpenCV's DNN module.

        This is the core "Transfer Learning" step of the project: rather
        than training a network from scratch, we load architecture and
        weights that were already trained on the PASCAL VOC dataset and
        reuse them directly for inference.

        Returns:
            A cv2.dnn_Net object ready to run forward-pass inference.

        Raises:
            ModelLoadError: If verification fails or OpenCV cannot parse
                the model files.
        """
        self.verify_model_files()

        try:
            logger.info("Loading MobileNet-SSD model into memory...")
            self.network = cv2.dnn.readNetFromCaffe(
                self.prototxt_path, self.weights_path
            )
            logger.info("MobileNet-SSD model loaded successfully.")
            return self.network
        except cv2.error as error:
            raise ModelLoadError(
                f"OpenCV failed to load the MobileNet-SSD model: {error}"
            ) from error

    def is_loaded(self) -> bool:
        """Return True if the model has already been loaded into memory."""
        return self.network is not None
