from ultralytics import YOLO
import cv2
import math 
import pyttsx3  # Import the text-to-speech library

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush","pen"
              ]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # List to store detected object names
    detected_objects = []

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])
            class_name = classNames[cls]
            detected_objects.append(class_name)

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, class_name, org, font, fontScale, color, thickness)

    # Convert detected object names to a single string
    detected_objects_str = ", ".join(detected_objects)
    
    # Speak out the detected object names
    engine.say(detected_objects_str)
    engine.runAndWait()

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Save the state_dict of the YOLO model
torch.save(model.model.state_dict(), "yolo_weights.pth")

import torch
from ultralytics import YOLO

# Instantiate the YOLO model
model = YOLO()

# Load the saved state_dict of the YOLO model
model.model.load_state_dict(torch.load("yolo_weights.pth"))

# Set the model to evaluation mode
model.eval()
