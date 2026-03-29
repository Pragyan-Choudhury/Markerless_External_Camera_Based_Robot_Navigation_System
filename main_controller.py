from IP_rtsp_stream_rotate import IPCamera
from yolodetect_botrob import YOLODetector
from track1 import Tracker
from localization_upd1 import Localizer
from map_builder import OccupancyGrid
from planner import AStarPlanner
from controller import PurePursuitController   # ✅ ADDED

import cv2
import time

# 🔷 GLOBAL VARIABLES
clicked_goal = None
latest_tracked_objects = []
localizer = None
grid_map = None

# 🔥 Planner globals
planner = None
path = None
prev_goal = None

# 🔥 Controller
controller = None   # ✅ ADDED


# 🔷 Mouse Click Callback
def mouse_callback(event, x, y, flags, param):
    global clicked_goal, latest_tracked_objects, localizer

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[INFO] Click at ({x}, {y})")

        for obj in latest_tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            cls = obj["class"]

            if x1 <= x <= x2 and y1 <= y <= y2:

                if cls == "robot":
                    print("[INFO] Robot clicked → ignored")
                    return

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                Xg, Yg = localizer.pixel_to_world(cx, cy)

                clicked_goal = {
                    "class": cls,
                    "pos": (round(Xg, 2), round(Yg, 2))
                }

                print(f"[GOAL SET] {cls} selected at {clicked_goal['pos']}")
                return


def main():
    global latest_tracked_objects, localizer, clicked_goal, grid_map
    global planner, path, prev_goal
    global controller   # ✅ ADDED

    ip_url = "http://192.168.0.106:8080/video"
    #ip_url = "http://172.20.10.12:8080/video"

    # 🔹 Camera
    camera = IPCamera(source=ip_url, width=640, height=480)
    camera.start()

    # 🔹 YOLO
    detector = YOLODetector(
        model_path="yolov8n.pt",
        conf=0.2,
        iou=0.5
    )

    # 🔹 Tracker
    tracker = Tracker()

    # 🔹 Localizer
    localizer = Localizer(
        frame_width=640,
        frame_height=480,
        world_width=4.0,
        world_height=3.0
    )

    # 🔹 Map
    grid_map = OccupancyGrid(
        width=4.0,
        height=3.0,
        resolution=0.1
    )

    # 🔹 Planner
    planner = AStarPlanner(grid_map)

    # 🔥 ✅ Controller Init
    controller = PurePursuitController(
        lookahead_dist=0.5,
        linear_speed=0.3,
        max_angular_speed=1.5
    )

    print("[INFO] Full Pipeline Running (Fixed Controller Integration)")

    INPUT_SIZE = 640
    prev_time = 0

    cv2.namedWindow("YOLO + Tracking Output")
    cv2.setMouseCallback("YOLO + Tracking Output", mouse_callback)

    try:
        while True:
            frame = camera.get_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            # 🔹 Resize
            frame_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

            # 🔹 Detection
            detections = detector.detect(frame_resized)

            # 🔹 Scale boxes back
            h, w, _ = frame.shape
            scale_x = w / INPUT_SIZE
            scale_y = h / INPUT_SIZE

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = [
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                ]

            # 🔹 Tracking
            try:
                tracked_objects = tracker.update(detections) if detections else []
            except Exception as e:
                print(f"[ERROR] Tracker failed: {e}")
                tracked_objects = []

            latest_tracked_objects = tracked_objects

            # 🔹 Localization
            robot_pose, obstacles = localizer.process(tracked_objects)

            # 🔹 Map
            occupancy_grid = grid_map.update(obstacles)

            # 🔹 Planning
            if robot_pose and clicked_goal:
                start = (robot_pose[0], robot_pose[1])
                goal = clicked_goal["pos"]

                if path is None or goal != prev_goal:
                    path = planner.plan(start, goal)
                    prev_goal = goal
                    print("[INFO] Path Planned")

                if obstacles:
                    path = planner.plan(start, goal)
                    print("[INFO] Replanned due to obstacles")

            # 🔥 ✅ CONTROLLER (ONLY ADDITION — SAFE)
            if robot_pose and path:
                v, w = controller.compute_control(robot_pose, path)
            else:
                v, w = 0.0, 0.0

            print(f"[CONTROL] v={v}, w={w}")

            # 🔹 Draw objects (⚠️ THIS WAS MISSING IN YOUR BUG CODE)
            for obj in tracked_objects:
                x1, y1, x2, y2 = obj["bbox"]
                cls = obj["class"]
                track_id = obj["id"]

                if cls == "robot":
                    color = (0, 0, 255)
                elif cls == "bottle":
                    color = (0, 255, 0)
                else:
                    color = (255, 255, 0)

                label = f"{cls} ID:{track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 🔹 Draw path
            if path:
                for i in range(len(path) - 1):
                    x1 = int((path[i][0] / localizer.world_width) * 640)
                    y1 = int((path[i][1] / localizer.world_height) * 480)

                    x2 = int((path[i+1][0] / localizer.world_width) * 640)
                    y2 = int((path[i+1][1] / localizer.world_height) * 480)

                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            # 🔹 Robot Pose
            if robot_pose:
                cv2.putText(frame,
                            f"R: ({robot_pose[0]}, {robot_pose[1]}, {robot_pose[2]})",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)
            # 🔹 Goal
            if clicked_goal:
                cv2.putText(frame,
                            f"GOAL: {clicked_goal['class']} {clicked_goal['pos']}",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2)
            # 🔹 FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # 🔹 Display control
            cv2.putText(frame,
                        f"v={v}, w={w}",
                        (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2)

            cv2.imshow("YOLO + Tracking Output", frame)
            # 🔷 DEBUG
            print(f"[DEBUG] FPS: {fps:.2f}")
            print(f"[DEBUG] Robot Pose: {robot_pose}")
            print(f"[DEBUG] Obstacles: {obstacles}")

            if path:
                print(f"[DEBUG] Path length: {len(path)}")

            if clicked_goal:
                print(f"[DEBUG] Goal: {clicked_goal['class']} at {clicked_goal['pos']}")
            else:
                print("[DEBUG] Goal: None")

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