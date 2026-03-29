from camera.rtsp_stream import Camera
from perception.yolo_detector import YOLODetector

import cv2
import time


def main():
    # 🔹 Step 1: Camera (webcam or RTSP)
    camera = Camera(source=0)
    camera.start()

    # 🔹 Step 2: YOLO Detector
    detector = YOLODetector(
        model_path="yolov8n.pt",
        conf=0.3,
        iou=0.5
    )

    print("[INFO] Starting YOLO Detection...")

    try:
        while True:
            frame = camera.get_frame()

            if frame is None:
                continue

            # 🔹 YOLO Detection
            detections = detector.detect(frame)

            # 🔹 Draw results
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cls = det["class"]
                conf = det["confidence"]

                label = f"{cls} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 🔹 Display
            cv2.imshow("YOLO Detection Output", frame)

            # 🔹 Debug output
            print(f"[DEBUG] Detections: {detections}")

            # Exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("[INFO] Program terminated")


if __name__ == "__main__":
    main()