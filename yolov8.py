from ultralytics import YOLO
import cv2


# model = YOLO("Weeds.pt")
model = YOLO("train/weights/best.pt")

result = model.predict(source="0", show = True, tracker="custom1.yaml")

print(result)