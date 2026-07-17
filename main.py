"""
main.py

The command-line entry point for the AI Object Detection System.

This file is intentionally kept "thin" — it does not contain any
detection logic, image processing, or file handling itself. Its only
job is to:

    1. Wire together the core components (ObjectDetector, DetectionHistory,
       AdminPanel).
    2. Present a simple text menu to the user.
    3. Collect and validate user input.
    4. Call the appropriate methods on the components above.
    5. Display results back to the user.

Run this file directly to start the application:
    python main.py
"""

import sys
import cv2

import config
from admin_panel import AdminPanel
from detection_history import DetectionHistory
from detector import ObjectDetector
from image_utils import InvalidImageError
from logger_setup import get_logger, log_detection_event
from model_loader import ModelLoadError

logger = get_logger(__name__)

MENU_TEXT = """
==================================================
   AI OBJECT DETECTION SYSTEM (MobileNet-SSD)
==================================================
1. Detect objects in an image
2. Change confidence threshold
3. View detection history
4. Clear detection history
5. Verify model files
6. Exit
==================================================
"""


def prompt_for_image_path() -> str:
    """Ask the user for an image path via the console."""
    return input("Enter the path to an image file: ").strip()


def prompt_for_threshold() -> float:
    """
    Ask the user for a confidence threshold and validate that it is a
    number within the allowed range, re-prompting on invalid input.
    """
    while True:
        raw_value = input(
            f"Enter confidence threshold "
            f"({config.MIN_CONFIDENCE_THRESHOLD}-{config.MAX_CONFIDENCE_THRESHOLD}, "
            f"default {config.DEFAULT_CONFIDENCE_THRESHOLD}): "
        ).strip()

        if raw_value == "":
            return config.DEFAULT_CONFIDENCE_THRESHOLD

        try:
            value = float(raw_value)
        except ValueError:
            print("Please enter a valid number (e.g. 0.5).")
            continue

        if not (config.MIN_CONFIDENCE_THRESHOLD <= value <= config.MAX_CONFIDENCE_THRESHOLD):
            print(
                f"Value must be between {config.MIN_CONFIDENCE_THRESHOLD} "
                f"and {config.MAX_CONFIDENCE_THRESHOLD}."
            )
            continue

        return value


def display_detection_results(result: dict) -> None:
    """
    Print a formatted detection summary to the console.

    Args:
        result: The dictionary returned by ObjectDetector.detect().
    """
    print("\nDetected Objects\n")

    if result["object_count"] == 0:
        print("No objects detected above the current confidence threshold.")
    else:
        for detection in result["detections"]:
            label = detection["label"].capitalize()
            confidence_percent = detection["confidence"] * 100
            print(f"{label} — {confidence_percent:.2f}%  |  Box: {detection['box']}")

    print(f"\nTotal Objects Detected: {result['object_count']}")


def handle_detect_objects(detector: ObjectDetector, history: DetectionHistory) -> None:
    """
    Run the "detect objects in an image" menu option end-to-end:
    prompt for input, run detection, display results, save output,
    and record history.
    """
    image_path = prompt_for_image_path()
    use_custom_threshold = input(
        "Use a custom confidence threshold for this run? (y/N): "
    ).strip().lower()

    if use_custom_threshold == "y":
        threshold = prompt_for_threshold()
        detector.set_confidence_threshold(threshold)

    try:
        result = detector.detect(image_path)
    except InvalidImageError as error:
        print(f"\nImage Error: {error}")
        return
    except RuntimeError as error:
        print(f"\nDetector Error: {error}")
        return

    display_detection_results(result)

# Display annotated image (resized if necessary)
    display_image = result["annotated_image"]

    # Maximum display size
    max_width = 1440
    max_height = 1440

    height, width = display_image.shape[:2]

    # Resize only if the image is larger than the display size
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)

        new_width = int(width * scale)
        new_height = int(height * scale)

        display_image = cv2.resize(
            display_image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    cv2.namedWindow("AI Object Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("AI Object Detection", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


        
    from image_utils import save_output_image
    output_path = save_output_image(result["annotated_image"], result["original_image_path"])
    print(f"\nAnnotated image saved to: {output_path}")

    history.add_entry(result["original_image_path"], result["detections"])
    log_detection_event(logger, image_path, result["object_count"])


def handle_change_threshold(admin: AdminPanel) -> None:
    """Run the "change confidence threshold" menu option."""
    print(f"Current threshold: {admin.get_current_threshold():.2f}")
    new_threshold = prompt_for_threshold()
    admin.change_confidence_threshold(new_threshold)
    print(f"Threshold updated to {new_threshold:.2f}")


def handle_view_history(admin: AdminPanel) -> None:
    """Run the "view detection history" menu option."""
    entries = admin.view_history()

    if not entries:
        print("No detection history yet.")
        return

    print(f"\nDetection History ({len(entries)} total entries)\n")
    for entry in entries:
        labels = ", ".join(entry["labels"]) if entry["labels"] else "none"
        print(
            f"[{entry['timestamp']}] {entry['image_name']} "
            f"-> {entry['object_count']} object(s): {labels}"
        )


def handle_clear_history(admin: AdminPanel) -> None:
    """Run the "clear detection history" menu option."""
    confirm = input("Are you sure you want to clear all history? (y/N): ").strip().lower()
    if confirm == "y":
        admin.clear_history()
        print("Detection history cleared.")
    else:
        print("Cancelled.")


def handle_verify_model(admin: AdminPanel) -> None:
    """Run the "verify model files" menu option."""
    try:
        admin.verify_model_files()
        print("Model files are present and valid.")
    except ModelLoadError as error:
        print(f"Model verification failed: {error}")


def run_application() -> None:
    """
    Initialize all core components and run the main interactive menu
    loop until the user chooses to exit.
    """
    print("Starting AI Object Detection System...")

    detector = ObjectDetector()
    history = DetectionHistory()
    admin = AdminPanel(detector, history)

    try:
        detector.load_model()
    except ModelLoadError as error:
        print(f"\nFATAL: Could not load the MobileNet-SSD model.\n{error}")
        sys.exit(1)

    print("Model loaded successfully. Ready to detect objects.\n")

    menu_actions = {
        "1": lambda: handle_detect_objects(detector, history),
        "2": lambda: handle_change_threshold(admin),
        "3": lambda: handle_view_history(admin),
        "4": lambda: handle_clear_history(admin),
        "5": lambda: handle_verify_model(admin),
    }

    while True:
        print(MENU_TEXT)
        choice = input("Select an option (1-6): ").strip()

        if choice == "6":
            print("Exiting AI Object Detection System. Goodbye!")
            break

        action = menu_actions.get(choice)
        if action is None:
            print("Invalid option. Please choose a number between 1 and 6.")
            continue

        action()


if __name__ == "__main__":
    run_application()
