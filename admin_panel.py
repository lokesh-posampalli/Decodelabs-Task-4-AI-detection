"""
admin_panel.py

A simple admin module that groups together administrative actions for
the AI Object Detection System:

    - Changing the confidence threshold
    - Viewing detection history
    - Clearing detection history
    - Verifying model files before loading

This module does not implement any of these features from scratch —
it composes functionality already provided by ObjectDetector,
DetectionHistory, and ModelLoader, and exposes it through a single,
convenient interface that main.py can call from a menu.
"""

import config
from detection_history import DetectionHistory
from detector import ObjectDetector
from logger_setup import get_logger
from model_loader import ModelLoader, ModelLoadError

logger = get_logger(__name__)


class AdminPanel:
    """
    Provides administrative operations on top of the core detection
    system components.

    An AdminPanel is given references to an existing ObjectDetector and
    DetectionHistory instance (rather than creating its own), so that
    changes made through the admin panel (like updating the confidence
    threshold) are immediately reflected in the same detector the user
    is actively using.
    """

    def __init__(self, detector: ObjectDetector, history: DetectionHistory) -> None:
        """
        Args:
            detector: The active ObjectDetector instance whose settings
                the admin panel can modify.
            history: The active DetectionHistory instance the admin
                panel can inspect or clear.
        """
        self.detector: ObjectDetector = detector
        self.history: DetectionHistory = history

    def change_confidence_threshold(self, new_threshold: float) -> None:
        """
        Update the confidence threshold used by the detector.

        Args:
            new_threshold: The new threshold value (0.0-1.0).

        Raises:
            ValueError: If the value is outside the valid range.
        """
        self.detector.set_confidence_threshold(new_threshold)
        logger.info(f"[Admin] Confidence threshold changed to {new_threshold:.2f}")

    def view_history(self, count: int | None = None) -> list[dict]:
        """
        Retrieve detection history entries.

        Args:
            count: If provided, return only the most recent `count`
                entries. If None, return the full history.

        Returns:
            A list of history entry dictionaries.
        """
        if count is None:
            return self.history.get_all_entries()
        return self.history.get_recent_entries(count)

    def clear_history(self) -> None:
        """Clear all recorded detection history."""
        self.history.clear_history()
        logger.info("[Admin] Detection history cleared.")

    def verify_model_files(self) -> bool:
        """
        Verify that the required MobileNet-SSD model files are present
        and valid, without actually loading the model into memory.

        Useful as a quick "health check" before a user attempts to run
        detection, e.g. right after cloning the project.

        Returns:
            True if the model files pass verification.

        Raises:
            ModelLoadError: If the files are missing or invalid, with a
                message explaining what's wrong.
        """
        checker = ModelLoader()
        result = checker.verify_model_files()
        logger.info("[Admin] Model file verification passed.")
        return result

    def get_current_threshold(self) -> float:
        """Return the confidence threshold currently in use by the detector."""
        return self.detector.confidence_threshold

    def get_history_count(self) -> int:
        """Return the total number of stored history entries."""
        return self.history.total_entries()
