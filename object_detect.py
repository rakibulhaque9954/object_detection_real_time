from ultralytics import YOLO
import cv2
import cvzone
import math

class_names = [
    "Person", "Bicycle", "Car", "Motorbike", "Aeroplane", "Bus", "Train", "Truck",
    "Boat", "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench",
    "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra",
    "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee", "Skis",
    "Snowboard", "Sports Ball", "Kite", "Baseball Bat", "Baseball Glove", "Skateboard",
    "Surfboard", "Tennis Racket", "Bottle", "Wine Glass", "Cup", "Fork", "Knife",
    "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli", "Carrot",
    "Hot Dog", "Pizza", "Donut", "Cake", "Chair", "Sofa", "Potted Plant", "Bed",
    "Dining Table", "Toilet", "TV Monitor", "Laptop", "Mouse", "Remote", "Keyboard",
    "Cell Phone", "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book",
    "Clock", "Vase", "Scissors", "Teddy Bear", "Hair Dryer", "Toothbrush"
]
model = YOLO("yolo-weights/yolov8n.pt")

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]  # bounding box coordinates which will return values as tensors
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # type casting tensors to int
            w, h = x2-x1, y2-y1
            print(x1, y1, w, h)
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil(box.conf[0] * 100) / 100
            print(conf)

            # Class Identification
            cls = box.cls[0]

            cvzone.putTextRect(img, f'{class_names[int(cls)]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
