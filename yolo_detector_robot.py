from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="yolov8m.pt", conf=0.3, iou=0.5, debug=False):
        """
        model_path: YOLO model (yolov8n, yolov8m, yolov8l)
        conf: confidence threshold
        iou: NMS IoU threshold
        debug: print all detected classes
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.debug = debug

        # ✅ Only detect chair (used as robot proxy)
        self.target_classes = ["chair"]

    def detect(self, frame):
        # 🔹 Run YOLO inference
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)

        detections = []

        for r in results:
            boxes = r.boxes

            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                class_name = self.model.names[cls_id]

                # 🔹 Debug: print everything YOLO detects
                if self.debug:
                    print(f"[DEBUG] Detected: {class_name} ({confidence:.2f})")

                # 🔹 Filter only chairs
                if class_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = map(int, xyxy)

                # 🔹 Optional: filter very small detections (noise)
                width = x2 - x1
                height = y2 - y1
                if width < 50 or height < 50:
                    continue

                # 🔹 Compute center (important for navigation)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "confidence": round(confidence, 2),
                    "class": "robot",   # 🔥 KEY: chair → robot
                    "id": None          # placeholder for tracking
                })

        # 🔴 Handle multiple chairs → select only ONE robot
        if len(detections) > 0:
            detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
            detections = [detections[0]]

        return detections