import sys
sys.path.append('c:/Users/galatta2/AppData/Local/Python/pythoncore-3.14-64/Lib/site-packages')
import cv2
from ultralytics import YOLO

# Load the object detection model (YOLOv8 nano is optimal for real-time CPU/GPU speed)
# For optimal accuracy, swap 'yolov8n.pt' with your custom-trained safety gear model path
model = YOLO('yolov8n.pt') 

# Initialize video capture (0 for default webcam, or provide a 'video.mp4' file path)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Starting stream. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the current frame
    # stream=True optimizes memory allocation for continuous video streams
    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract boundary coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract class ID and confidence score
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Fetch the human-readable label from model dictionary
            label = model.names[class_id]

            # Filter targets to track safety equipment or workers
            # (Adjust these class strings based on your specific custom dataset labels)
            if label in ['helmet', 'cell phone', 'person', 'safety-vest', 'bottle']:
                
                # Assign distinct bounding box colors based on compliance status
                if label in ['person']:
                    color = (0, 255, 0) # Green for protective gear
                elif label in ['cell phone']:
                    color = (0, 0, 255 )
                else:
                    color = (255, 0, 0) # Red for unverified individuals/objects

                # Draw bounding box rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Overlay classification text and confidence percentage
                text = f"{label} {confidence:.2f}"
                cv2.putText(frame, text, (x1, max(y1 - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Render frame in an interactive GUI window
    cv2.imshow('Real-Time OBJECT RECOGNITION ILLINOIS STATE FAIR', frame)

    # Break out of loop instantly if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Safely close video capture hardware and graphical windows
cap.release()
cv2.destroyAllWindows()
