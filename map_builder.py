import numpy as np


class OccupancyGrid:
    def __init__(self, width=4.0, height=3.0, resolution=0.1):
        """
        width, height → world size (meters)
        resolution → cell size (meters)
        """

        self.resolution = resolution

        self.cols = int(width / resolution)
        self.rows = int(height / resolution)

        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

    def world_to_grid(self, X, Y):
        gx = int(X / self.resolution)
        gy = int(Y / self.resolution)

        return gx, gy

    def update(self, obstacles):
        # Reset grid
        self.grid.fill(0)

        #for (X, Y) in obstacles:
        for obs in obstacles:
            X, Y = obs["pos"]
            
            gx, gy = self.world_to_grid(X, Y)

            if 0 <= gx < self.cols and 0 <= gy < self.rows:
                self.grid[gy][gx] = 1  # occupied

        return self.grid