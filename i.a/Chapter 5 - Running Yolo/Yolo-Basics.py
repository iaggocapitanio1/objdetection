from ultralytics import YOLO
import cv2

model = YOLO('../weights/yolov8l.pt')
results = model("Images/1.jpg", show=True)
cv2.waitKey(0)