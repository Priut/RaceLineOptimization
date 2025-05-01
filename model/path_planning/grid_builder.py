import numpy as np
import pygame
from matplotlib.path import Path

class GridBuilder:
    def __init__(self, width, height, resolution=5):
        """
        width, height: dimensions of the map in pixels
        resolution: size of each grid cell (in pixels)
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.cols = int(width // resolution)
        self.rows = int(height // resolution)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)  # shape: (rows, cols)

    def world_to_grid(self, x, y):
        """Convert from world coordinates (pixels) to grid indices."""
        i = int(x // self.resolution)
        j = int(y // self.resolution)
        # Clamp to bounds just in case
        i = max(0, min(i, self.cols - 1))
        j = max(0, min(j, self.rows - 1))
        return i, j

    def grid_to_world(self, i, j):
        """Convert from grid indices to world coordinates (center of cell)."""
        x = i * self.resolution + self.resolution // 2
        y = j * self.resolution + self.resolution // 2
        return x, y

    def mark_obstacles_from_track_edges(self, left_x, left_y, right_x, right_y):
        """Marks cells outside the track as obstacles based on track boundary points."""
        track_polygon = list(zip(left_x, left_y)) + list(zip(reversed(right_x), reversed(right_y)))
        path = Path(track_polygon)

        for j in range(self.rows):
            for i in range(self.cols):
                x, y = self.grid_to_world(i, j)
                if not path.contains_point((x, y)):
                    self.grid[j, i] = 1  # obstacle

    def is_valid_cell(self, i, j):
        return 0 <= i < self.cols and 0 <= j < self.rows and self.grid[j, i] == 0

    def get_cell_rects(self):
        """Return list of pygame.Rects for all grid cells."""
        rects = []
        for j in range(self.rows):
            for i in range(self.cols):
                x, y = self.grid_to_world(i, j)
                x -= self.resolution // 2
                y -= self.resolution // 2
                rect = pygame.Rect(x, y, self.resolution, self.resolution)
                rects.append((rect, self.grid[j, i]))  # 0 = free, 1 = obstacle
        return rects

