from object_names_list import class_names
from ultralytics import YOLO
import cv2
import cvzone
import math


model = YOLO("yolo-weights/yolov8n.pt")

class Video:
    def __init__(self, camera_index=0):
        self.video = cv2.VideoCapture(camera_index)
        self.video.set(3, 500)
        self.video.set(4, 700)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        results = model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]  # bounding box coordinates which will return values as tensors
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # type casting tensors to int
                w, h = x2 - x1, y2 - y1
                print(x1, y1, w, h)
                cvzone.cornerRect(frame, (x1, y1, w, h))

                # Confidence
                conf = math.ceil(box.conf[0] * 100) / 100
                print(conf)

                # Class Identification
                cls = box.cls[0]

                cvzone.putTextRect(frame, f'{class_names[int(cls)]} {conf}', (max(0, x1), max(35, y1)), scale=1,
                                   thickness=2)

        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()
