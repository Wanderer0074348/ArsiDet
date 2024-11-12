from ultralytics import YOLO
import cv2
import torch


model = YOLO('models/ArabicSignLanguage60.pt')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame, conf=0.30)

    for result in results:
        boxes = result.boxes

        if len(boxes) == 0:
            print("No detections found in the frame.")
        else:
            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0]

                confidence = box.conf[0]

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                print(f"Detected: {class_name}")
                print(f"Confidence: {confidence:.2f}")
                print(
                    f"Bounding Box: ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})")
                print("---")

    result_frame = results[0].plot()

    cv2.imshow('Arabic Sign Language Detection', result_frame)

    print(f"Frame shape: {frame.shape}")
    print(f"Inference time: {results[0].speed['inference']:.1f}ms")
    print(f"Preprocess time: {results[0].speed['preprocess']:.1f}ms")
    print(f"Postprocess time: {results[0].speed['postprocess']:.1f}ms")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()