import math


class Tracker:
    def __init__(self, iou_threshold=0.3, max_lost=10):
        self.next_id = 0
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def _get_new_id(self):
        self.next_id += 1
        return self.next_id

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)

        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        union_area = boxA_area + boxB_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def update(self, detections):
        updated_tracks = []
        used_track_ids = set()

        # 🔹 Match detections to tracks
        for det in detections:
            bbox = det["bbox"]
            cls = det["class"]

            best_iou = 0
            best_track = None

            for track in self.tracks:
                # ✅ IMPORTANT FIX: Match only same class
                if track["class"] != cls:
                    continue

                iou = self._iou(bbox, track["bbox"])

                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            # 🔹 If match found
            if best_iou > self.iou_threshold and best_track is not None:
                best_track["bbox"] = bbox
                best_track["lost"] = 0

                updated_tracks.append(best_track)
                used_track_ids.add(best_track["id"])

            else:
                # 🔹 Create new track
                new_track = {
                    "id": self._get_new_id(),
                    "bbox": bbox,
                    "class": cls,
                    "lost": 0
                }
                updated_tracks.append(new_track)
                used_track_ids.add(new_track["id"])

        # 🔹 Handle lost tracks
        for track in self.tracks:
            if track["id"] not in used_track_ids:
                track["lost"] += 1

                if track["lost"] <= self.max_lost:
                    updated_tracks.append(track)

        self.tracks = updated_tracks

        # 🔹 Final output
        output = []
        for track in self.tracks:
            output.append({
                "bbox": track["bbox"],
                "id": track["id"],
                "class": track["class"]
            })

        return output