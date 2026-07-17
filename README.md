# AI Object Detection System (MobileNet-SSD)

A beginner-friendly Computer Vision project that detects objects in images
using a pre-trained **MobileNet-SSD** model and **Transfer Learning** —
built as Project 4 of an AI Industrial Training Program.

---

## Project Overview

This project is a command-line AI Object Detection System. It loads a
MobileNet-SSD model that was already trained on the PASCAL VOC dataset,
runs it on a user-supplied image, and returns:

- Bounding boxes drawn around every detected object
- Object labels (e.g. "Person", "Car", "Dog")
- Confidence percentages for each detection
- A total object count
- A saved, annotated output image
- A persistent history of past detection runs

No model is trained from scratch in this project. The system relies
entirely on **Transfer Learning** — reusing a model someone else already
trained on a large dataset, and applying it directly for inference.

---

## Features

- Object detection on any image using pre-trained MobileNet-SSD
- Configurable confidence threshold (default `0.50`)
- Bounding box visualization with labels and confidence percentages
- Object counting per image
- Automatic saving of annotated output images
- Persistent detection history (JSON-backed)
- Simple admin module: change threshold, view/clear history, verify model files
- Colored console logging + persistent log file
- Input validation and custom exception handling throughout
- Unit tests covering the core pipeline (`unittest`)

---

## Technologies Used

| Technology       | Purpose                                      |
|-------------------|-----------------------------------------------|
| Python 3          | Core programming language                     |
| OpenCV (`cv2`)    | Image I/O, drawing, and DNN inference engine   |
| NumPy             | Array/data handling                            |
| MobileNet-SSD     | Pre-trained object detection model (Caffe)     |
| `unittest`        | Testing framework                              |
| `logging`         | Console + file logging                         |
| `json`            | Detection history persistence                  |

No OpenAI/Gemini APIs, no LLMs, no YOLO, no Detectron2, and no cloud
vision APIs are used anywhere in this project.

---

## Folder Structure

```
ai_object_detection/
│
├── config.py                  # Central configuration (paths, defaults, colors)
├── logger_setup.py            # Logging configuration (console + file, colored output)
├── model_loader.py            # Loads & verifies MobileNet-SSD model files
├── image_utils.py             # Image validation, bounding box drawing, saving
├── detector.py                # Core ObjectDetector engine (IPO pipeline)
├── detection_history.py       # Detection history tracking (save/view/clear)
├── admin_panel.py             # Admin module (threshold mgmt, history, model check)
├── main.py                    # CLI entry point — user interaction loop
│
├── tests/
│   └── test_detector.py       # unittest suite for the whole pipeline
│
├── models/                    # MobileNet-SSD weights, config, class labels (you add these)
├── images/                    # Input images go here
├── output/                    # Annotated output images are saved here
├── logs/                      # detection_log.txt and detection_history.json
│
├── requirements.txt
└── README.md
```

---

## Installation

**1. Clone or download this project**

```bash
git clone <your-repo-url>
cd ai_object_detection
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download the pre-trained MobileNet-SSD model files**

This project does not include the model weights (they're too large for a
typical repo). Download these two files and place them inside the
`models/` folder:

- `MobileNetSSD_deploy.prototxt` — the network architecture
- `MobileNetSSD_deploy.caffemodel` — the pre-trained weights

Both files are widely available from public MobileNet-SSD / OpenCV
sample-model repositories. Once downloaded, your `models/` folder should
contain exactly these two files.

**4. Verify the model files (optional but recommended)**

Run the app and choose option `5. Verify model files` from the menu, or
run the test suite — tests that need the model are skipped automatically
if the files aren't present yet.

---

## Execution Steps

Run the application from the project root:

```bash
python main.py
```

You'll see a menu:

```
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
```

Run the test suite:

```bash
python -m unittest tests/test_detector.py -v
```

---

## Example User Interaction

```
Select an option (1-6): 1
Enter the path to an image file: images/street.jpg
Use a custom confidence threshold for this run? (y/N): n

Detected Objects

Person — 98.43%  |  Box: (120, 45, 300, 480)
Bottle — 92.15%  |  Box: (410, 220, 460, 350)
Laptop — 95.67%  |  Box: (50, 300, 220, 420)

Total Objects Detected: 3

Annotated image saved to: output/street_detected_20260716_141032.jpg
```

---

## IPO Workflow (Input → Process → Output)

This project is structured around the IPO model:

**Input**
- An image file path provided by the user
- A confidence threshold (default `0.50`, adjustable)

**Process**
1. Validate the image path and file extension
2. Load the pre-trained MobileNet-SSD model (once, at startup)
3. Convert the image into a normalized "blob" and run inference
4. Filter raw detections using the confidence threshold
5. Map class IDs to human-readable labels
6. Draw bounding boxes and labels on a copy of the image
7. Count total detected objects
8. Record the run in detection history

**Output**
- An annotated image saved to `output/`
- A console detection summary (labels, confidence, box coordinates)
- An updated detection history entry
- Log entries in `logs/detection_log.txt`

---

## Transfer Learning Explanation

Transfer Learning means reusing a model that was already trained on a
large dataset, instead of training a new one from scratch.

MobileNet-SSD used in this project was originally trained on the
**PASCAL VOC dataset**, which contains thousands of labeled images across
20 object categories (person, car, dog, bottle, chair, and more). That
training process — learning to recognize shapes, edges, and object
patterns — already happened before this project even started.

This project simply **loads those pre-trained weights** and applies the
model directly to new images (a process called *inference*). No
training, backpropagation, or dataset collection happens here — which is
exactly what makes MobileNet-SSD lightweight and practical for a
one-week beginner project.

---

## Bounding Box Explanation

A bounding box is the rectangle drawn around a detected object, defined
by two corner points: `(x1, y1)` (top-left) and `(x2, y2)` (bottom-right).

MobileNet-SSD outputs box coordinates as **normalized values** between 0
and 1 (relative to image width/height). This project converts them back
into real pixel coordinates using the original image's dimensions, then
draws:

- A colored rectangle around the object
- A label showing the object name
- The confidence percentage next to the label
- A small filled background behind the text so it stays readable over
  busy or bright image regions

---

## Confidence Threshold Explanation

Every detection MobileNet-SSD produces comes with a confidence score
between 0 and 1 (0% to 100%), representing how certain the model is that
it found a real object of that class.

The confidence threshold is a cutoff value — any detection scoring below
it is discarded. For example:

- Threshold `0.50` → keeps detections the model is at least 50% sure about
- Threshold `0.90` → keeps only very high-confidence detections, reducing
  false positives but possibly missing some real objects
- Threshold `0.20` → keeps almost everything, including weaker guesses

This project defaults to `0.50` and lets the user adjust it per run or
permanently through the admin menu.

---

## Future Enhancements

- Real-time detection from a webcam feed
- Batch processing of an entire folder of images
- Support for additional pre-trained SSD/MobileNet variants
- A simple web interface (Flask) instead of CLI-only interaction
- Exporting detection history to CSV for reporting
- Non-Maximum Suppression tuning for overlapping boxes

---

## Author's Note

This project was built as Project 4 of a one-week AI Industrial Training
Program, focused on applying Transfer Learning and Computer Vision
fundamentals using OpenCV — without training any custom model from
scratch.
