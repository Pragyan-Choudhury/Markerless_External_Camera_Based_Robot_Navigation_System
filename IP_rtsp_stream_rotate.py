import cv2
import time
import threading


class IPCamera:
    def __init__(self, source, width=640, height=480):
        """
        source:
            - Phone IP camera URL → "http://192.168.x.x:8080/video"
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
        print(f"[INFO] Connecting to IP Camera: {self.source}")

        # Use FFMPEG backend for better streaming support
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            print("[ERROR] Failed to connect. Retrying in 2 sec...")
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
            if not self.cap.isOpened():
                print("[WARNING] Capture not open. Reconnecting...")
                self._connect()
                continue

            ret, frame = self.cap.read()

            if not ret or frame is None:
                print("[WARNING] Frame not received. Reconnecting...")
                self.cap.release()
                self._connect()
                continue

            # 🔹 STEP 1: Rotate 90° clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # 🔹 STEP 2: Resize
            frame = cv2.resize(frame, (self.width, self.height))

            # Store safely
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