import cv2
import time
import threading


class Camera:
    def __init__(self, source=0, width=640, height=480):
        """
        source:
            - RTSP URL → "rtsp://..."
            - Webcam → 0, 1, etc.
        """
        self.source = source
        self.width = width
        self.height = height

        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False

        self._connect()

    def _connect(self):
        print(f"[INFO] Connecting to source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print("[ERROR] Failed to connect. Retrying...")
            time.sleep(2)
            self._connect()
        else:
            print("[INFO] Connected successfully")

    def start(self):
        if self.running:
            return

        self.running = True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                print("[WARNING] Frame not received. Reconnecting...")
                self.cap.release()
                self._connect()
                continue

            frame = cv2.resize(frame, (self.width, self.height))

            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        print("[INFO] Camera stopped")