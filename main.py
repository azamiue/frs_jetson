import cv2
import torch
import time
import platform
from mtcnn.mtcnn import MTCNN

# Lưu log vào file
log_file = "benchmark_log.txt"

# Khởi tạo MTCNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# Đọc video thay vì dùng webcam
video_path = r"C:\Users\thainq\Desktop\capstone\frs_jetson\vid_test\test.mp4"  # Thay bằng đường dẫn tới video của bạn
cap = cv2.VideoCapture(video_path)

# Biến lưu trữ thông số benchmark
frame_count = 0
total_time = 0
fps_list = []
face_count_list = []
inference_times = []

# Lấy thông tin hệ thống
system_info = f"Device: {device}, System: {platform.system()} {platform.release()}, Processor: {platform.processor()}"
print(system_info)

# Ghi thông tin vào file log
with open(log_file, "w") as f:
    f.write(system_info + "\n")
    f.write("Frame, FPS, Processing Time (s), Inference Time (s), Faces Detected\n")

# Thêm lựa chọn cho người dùng
option = input("Chọn chế độ (1: No Limit, 2: Limit 200 frames): ")
if option == "1":
    limit = float('inf')  # Không giới hạn
elif option == "2":
    limit = 200  # Giới hạn 200 frame
else:
    print("Lựa chọn không hợp lệ. Mặc định là 200 frames.")
    limit = 200

while frame_count < limit:  # Chạy tùy thuộc vào lựa chọn
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1  # Đếm số frame đã xử lý

    # Chuyển đổi BGR (OpenCV) sang RGB (MTCNN)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Đo thời gian chạy mô hình (inference)
    start_infer = time.time()
    boxes, _ = mtcnn.detect(rgb_frame)
    inference_time = time.time() - start_infer
    inference_times.append(inference_time)

    face_count = len(boxes) if boxes is not None else 0
    face_count_list.append(face_count)

    # Vẽ bounding box nếu có khuôn mặt
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Tính thời gian xử lý và FPS
    processing_time = time.time() - start_time
    total_time += processing_time
    fps = 1 / processing_time if processing_time > 0 else 0
    fps_list.append(fps)

    # Hiển thị FPS và Inference Time trên màn hình
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Infer Time: {inference_time:.4f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Hiển thị video
    cv2.imshow('Face Detection - MTCNN', frame)

    # Ghi log vào file
    with open(log_file, "a") as f:
        f.write(f"{frame_count}, {fps:.2f}, {processing_time:.4f}, {inference_time:.4f}, {face_count}\n")

    # Nhấn 'q' để thoát sớm
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tính toán thông số tổng kết
avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
avg_faces = sum(face_count_list) / len(face_count_list) if face_count_list else 0
avg_infer_time = sum(inference_times) / len(inference_times) if inference_times else 0

summary = f"\nBenchmark Summary:\nTotal Frames: {frame_count}\nAverage FPS: {avg_fps:.2f}\nAverage Faces Detected: {avg_faces:.2f}\nAverage Inference Time: {avg_infer_time:.4f} sec\nTotal Processing Time: {total_time:.2f} sec\n"
print(summary)

# Ghi thông số tổng kết vào log
with open(log_file, "a") as f:
    f.write(summary)

cap.release()
cv2.destroyAllWindows()
