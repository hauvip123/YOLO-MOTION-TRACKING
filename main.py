import cv2
from ultralytics import YOLO
model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, imgsz=640, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("Pose Estimation", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
