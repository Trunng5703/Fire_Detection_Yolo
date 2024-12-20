import sys
import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO('FireDetectionModel.pt')  # Đường dẫn đến mô hình đã huấn luyện lại

while True:
    success, im = cap.read() # Lấy ảnh từ camera
    if not success:
        break

    # Sử dụng mô hình đã huấn luyện lại để phát hiện đối tượng
    results = model.track(im)

    # Duyệt qua các kết quả phát hiện và vẽ bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Trích xuất tọa độ bounding box từ tensor
            conf = box.conf[0].item() # Trích xuất độ tin cậy từ tensor
            cls = int(box.cls[0]) # Trích xuất class ID từ tensor

            # Chỉ vẽ bounding box khi độ tin cậy >= 0.8
            if conf >= 0.8:
                # Vẽ bounding box lên ảnh
                cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Hiển thị tên lớp và độ tin cậy theo phần trăm
                label = f"{result.names[cls]} {conf * 100:.2f}%"
                cv2.putText(im, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow('Image', im) # Hiển thị ảnh với các bounding box
    if cv2.waitKey(1) & 0xFF == ord('q'): # Nhấn phím 'q' để thoát
        break

    # Thu dọn bộ nhớ sau mỗi vòng lặp
    im = None
    results = None
    gc.collect()

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

# Thêm dòng này để chắc chắn tiến trình Python được giải phóng hoàn toàn
import os
os._exit(0)
