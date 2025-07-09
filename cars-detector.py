import cv2
from ultralytics import YOLO

# 1. Load YOLOv8 model
model = YOLO("yolov8n.pt")

# 2. Open the video
cap = cv2.VideoCapture("cars.mp4")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run inference
    results = model(frame, verbose=False)[0]

    # 4. Draw boxes for class “car” (COCO class id 2)
    for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                              results.boxes.cls.cpu().numpy(),
                              results.boxes.conf.cpu().numpy()):
        if int(cls) == 2:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"car {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

    cv2.imshow("Car Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()