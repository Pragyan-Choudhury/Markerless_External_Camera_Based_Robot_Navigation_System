import math


class Localizer:
    def __init__(self, frame_width=640, frame_height=480,
                 world_width=4.0, world_height=3.0):

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.world_width = world_width
        self.world_height = world_height

        # Store previous robot position (for theta)
        self.prev_robot_pos = None

    def pixel_to_world(self, cx, cy):
        """
        Convert pixel → world coordinates
        """
        X = (cx / self.frame_width) * self.world_width
        Y = (cy / self.frame_height) * self.world_height

        return X, Y

    def compute_theta(self, prev, curr):
        """
        Compute orientation (theta)
        """
        if prev is None:
            return 0.0

        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]

        if dx == 0 and dy == 0:
            return 0.0

        theta = math.atan2(dy, dx)
        return theta

    def process(self, tracked_objects):
        """
        Input: tracker output

        Output:
            robot_pose = (X, Y, theta)
            obstacles = [
                {"id": id, "class": cls, "pos": (X, Y)},
                ...
            ]
        """

        robot_pose = None
        obstacles = []

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            cls = obj["class"]
            track_id = obj.get("id", None)

            # 🔹 center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # 🔹 convert to world
            X, Y = self.pixel_to_world(cx, cy)

            if cls == "robot":
                curr_pos = (X, Y)
                theta = self.compute_theta(self.prev_robot_pos, curr_pos)

                robot_pose = (round(X, 2), round(Y, 2), round(theta, 2))
                self.prev_robot_pos = curr_pos

            else:
                obstacles.append({
                    "id": track_id,
                    "class": cls,
                    "pos": (round(X, 2), round(Y, 2))
                })

        return robot_pose, obstacles