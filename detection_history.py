"""
detection_history.py

Manages a persistent history of past detection runs.

Every time the user detects objects in an image, a summary entry
(timestamp, image name, object count, detected labels) is appended to
a JSON file on disk. This lets the admin module and the user review
what was processed recently, without needing a database.
"""

import json
import os
from datetime import datetime

import config
from logger_setup import get_logger

logger = get_logger(__name__)


class DetectionHistory:
    """
    Reads and writes a JSON-backed history of detection runs.

    The history is capped at config.MAX_HISTORY_ENTRIES entries; the
    oldest entries are dropped first once that limit is reached, so the
    history file doesn't grow indefinitely over a long training week.
    """

    def __init__(self, history_file_path: str = config.HISTORY_FILE_PATH) -> None:
        """
        Args:
            history_file_path: Path to the JSON file used to persist
                history entries between runs.
        """
        self.history_file_path: str = history_file_path
        self._entries: list[dict] = self._load_from_disk()

    def _load_from_disk(self) -> list[dict]:
        """
        Load existing history entries from the JSON file, if present.

        Returns:
            A list of history entry dicts. Returns an empty list if the
            file doesn't exist yet or contains invalid JSON (rather than
            crashing the whole application over a corrupted history file).
        """
        if not os.path.isfile(self.history_file_path):
            return []

        try:
            with open(self.history_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    return data
                logger.warning("History file did not contain a list; starting fresh.")
                return []
        except (json.JSONDecodeError, OSError) as error:
            logger.warning(f"Could not read history file ({error}); starting fresh.")
            return []

    def _save_to_disk(self) -> None:
        """Persist the current in-memory entries list to the JSON file."""
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        with open(self.history_file_path, "w", encoding="utf-8") as file:
            json.dump(self._entries, file, indent=2)

    def add_entry(self, image_path: str, detections: list[dict]) -> None:
        """
        Record a new detection run in the history.

        Args:
            image_path: Path of the image that was processed.
            detections: The list of detection dicts produced by
                ObjectDetector.detect() (each with 'label', 'confidence').
        """
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_name": os.path.basename(image_path),
            "object_count": len(detections),
            "labels": [detection["label"] for detection in detections],
        }

        self._entries.append(entry)

        # Trim oldest entries if we've exceeded the configured cap.
        if len(self._entries) > config.MAX_HISTORY_ENTRIES:
            self._entries = self._entries[-config.MAX_HISTORY_ENTRIES:]

        self._save_to_disk()
        logger.info(f"History entry added for '{entry['image_name']}'.")

    def get_all_entries(self) -> list[dict]:
        """Return all recorded history entries, oldest first."""
        return list(self._entries)

    def get_recent_entries(self, count: int = 5) -> list[dict]:
        """
        Return the most recent history entries.

        Args:
            count: Number of recent entries to return.

        Returns:
            A list of the last `count` entries, most recent last.
        """
        return self._entries[-count:]

    def clear_history(self) -> None:
        """Erase all recorded history entries, both in memory and on disk."""
        self._entries = []
        self._save_to_disk()
        logger.info("Detection history cleared.")

    def total_entries(self) -> int:
        """Return the total number of history entries currently stored."""
        return len(self._entries)
