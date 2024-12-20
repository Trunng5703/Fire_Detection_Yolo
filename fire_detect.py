import sys
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Load model một lần ở global scope
model = YOLO("./FireDetectionModel.pt")
annotator = sv.BoxAnnotator()

def process_frame(frame):
    try:
        # Resize ảnh để tăng tốc độ xử lý
        frame = cv2.resize(frame, (640, 480))

        # Thực hiện detect
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        frame = annotator.annotate(scene=frame, detections=detections)

        return frame
    except Exception as e:
        sys.stderr.write(f"Error: {str(e)}\n")
        return None

if __name__ == "__main__":
    try:
        # Mở webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Không thể mở webcam")

        while True:
            # Đọc frame từ webcam
            ret, frame = cap.read()
            if not ret:
                sys.stderr.write("Không thể đọc frame từ webcam\n")
                break

            # Xử lý frame
            processed_frame = process_frame(frame)
            if processed_frame is None:
                sys.stderr.write("Không thể xử lý frame\n")
                break

            # Hiển thị frame đã xử lý
            cv2.imshow("Fire Detection", processed_frame)

            # Dừng lại khi nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        sys.stderr.write(f"Error: {str(e)}\n")
        sys.exit(1)