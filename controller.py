import math


class PurePursuitController:
    def __init__(self,
                 lookahead_dist=0.5,
                 linear_speed=0.3,
                 max_angular_speed=1.5):
        """
        lookahead_dist → how far ahead to track (meters)
        linear_speed → base forward velocity (m/s)
        max_angular_speed → clamp turning speed (rad/s)
        """

        self.lookahead_dist = lookahead_dist
        self.linear_speed = linear_speed
        self.max_angular_speed = max_angular_speed

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def find_lookahead_point(self, robot_pos, path):
        """
        Find first point in path at lookahead distance
        """
        for point in path:
            if self.distance(robot_pos, point) >= self.lookahead_dist:
                return point

        return path[-1]  # fallback to goal

    def compute_control(self, robot_pose, path):
        """
        Input:
            robot_pose = (x, y, theta)
            path = [(x1,y1), ...]

        Output:
            v (linear velocity)
            w (angular velocity)
        """

        if not path or robot_pose is None:
            return 0.0, 0.0

        rx, ry, theta = robot_pose

        # 🔹 Get lookahead target
        target = self.find_lookahead_point((rx, ry), path)
        tx, ty = target

        # 🔹 Compute angle to target
        angle_to_target = math.atan2(ty - ry, tx - rx)

        # 🔹 Angle difference
        alpha = angle_to_target - theta

        # Normalize angle [-pi, pi]
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # 🔹 Control law
        v = self.linear_speed
        w = 2 * v * math.sin(alpha) / self.lookahead_dist

        # 🔹 Clamp angular velocity
        w = max(min(w, self.max_angular_speed), -self.max_angular_speed)

        # 🔹 Slow down near goal
        if self.distance((rx, ry), path[-1]) < 0.2:
            v = 0.0
            w = 0.0

        return round(v, 3), round(w, 3)