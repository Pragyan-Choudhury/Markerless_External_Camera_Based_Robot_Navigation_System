from rtsp_stream_webcam import Camera
import cv2
import time


def main():
    # 🔹 INPUT: Webcam (0 = default webcam)
    camera = Camera(source=0, width=640, height=480)

    # Start camera thread
    camera.start()

    print("[INFO] Starting webcam frame ingestion...")

    try:
        while True:
            # 🔹 OUTPUT: Frame (numpy array)
            frame = camera.get_frame()

            if frame is None:
                continue

            # --- Visualization ---
            cv2.imshow("Webcam Stream - Step 1 Output", frame)

            # Debug info
            print(f"[DEBUG] Frame shape: {frame.shape}")

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Optional delay (reduce CPU usage)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        print("[INFO] Program terminated")


if __name__ == "__main__":
    main()