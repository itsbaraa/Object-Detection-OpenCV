# Object Detection with OpenCV and YOLOv8
A Python project that detects cars in a video using the Ultralytics YOLOv8 model and displays the results live with OpenCV.

## Prerequisites

- Python 3.8+

## Install dependencies
```bash
pip install ultralytics opencv-python
```

The code will:
1. Load the tiny YOLOv8 model (`yolov8n.pt`).
2. Read frames from `cars.mp4`.
3. Draw green bounding-boxes around objects classified as **car**.
4. Show the annotated frames in a window until you press `Esc`.

## Showcase
![Showcase](https://github.com/user-attachments/assets/0b10767b-65f1-466e-b21d-cccb645147c3)
