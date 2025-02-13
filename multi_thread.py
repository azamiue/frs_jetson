import cv2
import torch
import time
import platform
import threading
import queue
from mtcnn.mtcnn import MTCNN

# Lưu log vào file
log_file = "benchmark_log_multi_thread.txt"

# Khởi tạo MTCNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# Mở webcam
cap = cv2.VideoCapture(0)

# Biến lưu trữ thông số benchmark
frame_count = 0
total_time = 0
fps_list = []
face_count_list = []
inference_times = []

# Hàng đợi để xử lý frame
frame_queue = queue.Queue()
output_queue = queue.Queue()

def capture_frames():
    """Thread 1: Capture frames liên tục"""
    global frame_count
    while frame_count < 200:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
        frame_count += 1
    cap.release()

def process_frames():
    """Thread 2: Xử lý frames với MTCNN"""
    global total_time
    while frame_count < 200 or not frame_queue.empty():
        if not frame_queue.empty():
            frame = frame_queue.get()
            start_time = time.time()

            # Chuyển đổi BGR (OpenCV) sang RGB (MTCNN)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Đo thời gian chạy inference
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

            # Ghi log vào file
            with open(log_file, "a") as f:
                f.write(f"{frame_count}, {fps:.2f}, {processing_time:.4f}, {inference_time:.4f}, {face_count}\n")

            output_queue.put((frame, fps, inference_time))

def display_frames():
    """Thread 3: Hiển thị frames"""
    while frame_count < 200 or not output_queue.empty():
        if not output_queue.empty():
            frame, fps, inference_time = output_queue.get()
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Infer Time: {inference_time:.4f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Face Detection - MTCNN', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Chạy các thread song song
t1 = threading.Thread(target=capture_frames)
t2 = threading.Thread(target=process_frames)

t1.start()
t2.start()

t1.join()
t2.join()

# Tính toán thông số tổng kết
avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
avg_faces = sum(face_count_list) / len(face_count_list) if face_count_list else 0
avg_infer_time = sum(inference_times) / len(inference_times) if inference_times else 0

summary = f"\nBenchmark Summary:\nTotal Frames: {frame_count}\nAverage FPS: {avg_fps:.2f}\nAverage Faces Detected: {avg_faces:.2f}\nAverage Inference Time: {avg_infer_time:.4f} sec\nTotal Processing Time: {total_time:.2f} sec\n"
print(summary)

# Ghi thông số tổng kết vào log
with open(log_file, "a") as f:
    f.write(summary)
