import heapq


class AStarPlanner:
    def __init__(self, grid_map):
        self.grid_map = grid_map

    def heuristic(self, a, b):
        # Euclidean distance
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def get_neighbors(self, node):
        x, y = node

        # 8-direction movement
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, -1), (1, -1), (-1, 1)
        ]

        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < self.grid_map.cols and 0 <= ny < self.grid_map.rows:
                # Check obstacle
                if self.grid_map.grid[ny][nx] == 0:
                    neighbors.append((nx, ny))

        return neighbors

    def plan(self, start_world, goal_world):
        """
        start_world = (X, Y)
        goal_world = (X, Y)
        """

        # Convert to grid
        start = self.grid_map.world_to_grid(*start_world)
        goal = self.grid_map.world_to_grid(*goal_world)

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g

                    f_score = tentative_g + self.heuristic(neighbor, goal)

                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current

        return None  # No path found

    def reconstruct_path(self, came_from, current):
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        # Convert grid → world
        world_path = []
        for gx, gy in path:
            X = gx * self.grid_map.resolution
            Y = gy * self.grid_map.resolution
            world_path.append((round(X, 2), round(Y, 2)))

        return world_path