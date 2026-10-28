# StreetVision AI

A web-based object detection application built with Python, Streamlit, and YOLO. The application allows users to upload street images, detect objects, and review the results through visualizations and structured detection data.

## Overview

StreetVision AI applies a pretrained YOLO model to identify objects in street scenes. It displays the detected objects using bounding boxes and confidence scores, along with a summary of the detection results.

The application is designed to provide a simple interface for exploring object detection without requiring users to interact directly with the model through Python code.

## Features

- Upload images for object detection.
- Visualize detected objects with bounding boxes and confidence scores.
- Adjust the confidence threshold.
- View object counts by category.
- Review detection results in a structured table.
- Display model inference time.
- Download annotated images and detection results in CSV format.

## Technology Stack

- **Python** — application logic
- **Streamlit** — web interface
- **Ultralytics YOLO** — object detection
- **Pandas** — detection data processing
- **Pillow** — image handling
- **Plotly** — data visualization

## Model

The application is configured to use the pretrained `yolov8n.pt` model from Ultralytics.

YOLOv8 Nano is a lightweight object detection model suitable for applications where inference speed and computational efficiency are important.

The model's predictions depend on image quality, object visibility, object size, and the selected confidence threshold.

## Getting Started

### Prerequisites

- Python
- pip
- An internet connection to download model weights if they are not available locally

### Installation

Clone the repository:

```bash
git clone https://github.com/RahulLakeser/StreetVision-YOLO.git
```

Navigate to the project directory:

```bash
cd StreetVision-YOLO
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will provide a local URL where the application can be accessed in a web browser.

## Project Structure

```text
StreetVision-YOLO/
├── app.py
├── requirements.txt
└── README.md
```

## Workflow

1. Upload a street image through the application.
2. Run object detection using the configured YOLO model.
3. View the annotated image and predicted object categories.
4. Review confidence scores, object counts, and inference time.
5. Download the annotated image or export the detection results.

## Limitations

Detection results are predictions and may not always be accurate. Performance can vary depending on image resolution, lighting, occlusion, object size, and available computing resources.

## Author

**Rahul Lakeser**

[GitHub Profile](https://github.com/RahulLakeser)

## License

No license has been specified for this repository yet.
