from ultralytics import YOLO
import cv2
import torch

# Load the model
model = YOLO('ArabicSignLanguage60.pt')

# Ensure CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Access the system camera
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference
    results = model(frame, conf=0.25)  # Lower confidence threshold

    # Process and print the results
    for result in results:
        boxes = result.boxes  # Bounding box outputs
        
        if len(boxes) == 0:
            print("No detections found in the frame.")
        else:
            for box in boxes:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                
                # Get the confidence score
                confidence = box.conf[0]
                
                # Get the class label
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                print(f"Detected: {class_name}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Bounding Box: ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})")
                print("---")

    # Visualize the results
    result_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('Arabic Sign Language Detection', result_frame)

    # Print inference details
    print(f"Frame shape: {frame.shape}")
    print(f"Inference time: {results[0].speed['inference']:.1f}ms")
    print(f"Preprocess time: {results[0].speed['preprocess']:.1f}ms")
    print(f"Postprocess time: {results[0].speed['postprocess']:.1f}ms")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()