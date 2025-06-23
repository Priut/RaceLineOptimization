import heapq
import math
import numpy as np
from scipy.interpolate import splev
from model.path_planning.node import Node


class AStarPlanner:
    """
    A* path planner.
    Considers physical constraints and racing objectives:
    - Time-optimal movement based on curvature and car dynamics
    - Clockwise direction bias
    - Alignment with centerline (heading angle)
    - Proximity to centerline (lateral offset)
    """

    def __init__(self, grid_builder, sim_car, center_spline=None, logging=False):
        """
        Parameters:
            grid_builder (GridBuilder): Map with obstacle grid and resolution
            sim_car (Car): Car physics model to compute speed from curvature
            center_spline (tuple): Spline (tck) representation of centerline for angle/lateral costs
        """
        self.grid = grid_builder.grid
        self.grid_builder = grid_builder
        self.cols = grid_builder.cols
        self.rows = grid_builder.rows
        self.sim_car = sim_car
        self.center_spline = center_spline
        self.nodes_expanded = 0
        self.logging=logging
        self.log_file = "astar_cost_log.csv"

        # Log header for debugging and tuning
        if self.logging:
            with open(self.log_file, "w") as f:
                f.write("from_x,from_y,to_x,to_y,time_cost,direction_penalty,angle_penalty,lateral_penalty,total_cost\n")

    def heuristic(self, node, goal):
        """Euclidean distance heuristic."""
        dx = goal.x - node.x
        dy = goal.y - node.y
        return math.hypot(dx, dy)

    def get_neighbors(self, node):
        """Return 8-connected grid neighbors if they are within bounds and not obstacles."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = node.x + dx, node.y + dy
            if self.grid_builder.is_valid_cell(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def cost(self, from_node, to_node):
        """
        Compute the cost of moving from one node to another based on:
        - Time (computed using car dynamics + curvature)
        - Clockwise direction preference
        - Heading angle relative to track direction
        - Lateral deviation from track center
        """
        from_world = self.grid_builder.grid_to_world(from_node.x, from_node.y)
        to_world = self.grid_builder.grid_to_world(to_node.x, to_node.y)

        dx = to_world[0] - from_world[0]
        dy = to_world[1] - from_world[1]
        distance = math.hypot(dx, dy)

        # Estimate curvature radius
        radius = float('inf')
        if from_node.parent and from_node.parent.parent:
            p2 = self.grid_builder.grid_to_world(from_node.parent.parent.x, from_node.parent.parent.y)
            p1 = self.grid_builder.grid_to_world(from_node.parent.x, from_node.parent.y)
            p3 = from_world
            p4 = to_world
            radius = self._compute_radius_extended(p2, p1, p3, p4)
        elif from_node.parent:
            p1 = self.grid_builder.grid_to_world(from_node.parent.x, from_node.parent.y)
            radius = self._compute_radius(p1, from_world, to_world)

        # Compute speed and time
        max_speed = self.sim_car.compute_max_lateral_speed(radius)
        max_speed = max(1.0, min(max_speed, 40.0))
        time_cost = distance / max_speed

        # Penalize counter-clockwise motion
        cx, cy = self.cols // 2, self.rows // 2
        vec_from = [from_node.x - cx, from_node.y - cy]
        vec_to = [to_node.x - cx, to_node.y - cy]
        cross = vec_from[0] * vec_to[1] - vec_from[1] * vec_to[0]
        direction_penalty = 1 if cross > 0 else 0

        angle_penalty = 0
        lateral_penalty = 0

        if self.center_spline:
            wx, wy = to_world

            u_fine = np.linspace(0, 1, 1000)
            cx_array, cy_array = splev(u_fine, self.center_spline)
            dists = (cx_array - wx) ** 2 + (cy_array - wy) ** 2
            u_closest = u_fine[np.argmin(dists)]

            # Angle to centerline tangent
            dx_dt, dy_dt = splev(u_closest, self.center_spline, der=1)
            tangent = np.array([dx_dt, dy_dt])
            tangent /= np.linalg.norm(tangent) + 1e-8

            heading = np.array([dx, dy], dtype=np.float64)
            heading /= np.linalg.norm(heading) + 1e-8

            dot = np.clip(np.dot(heading, tangent), -1, 1)
            angle_penalty = (1 - dot)  # dot=1 -> aligned

            # Lateral distance to centerline
            center_pos = np.array([cx_array[np.argmin(dists)], cy_array[np.argmin(dists)]])
            dist_to_centerline = np.linalg.norm(np.array([wx, wy]) - center_pos)
            lateral_penalty = dist_to_centerline

        # Final cost
        angle_weight = 1
        lateral_weight = 1
        direction_weight = 100
        total_cost = (time_cost +
                      direction_weight * direction_penalty +
                      angle_weight * angle_penalty +
                      lateral_weight * lateral_penalty)

        if self.logging:
            with open(self.log_file, "a") as f:
                f.write(f"{from_node.x},{from_node.y},{to_node.x},{to_node.y},"
                        f"{time_cost:.4f},{direction_penalty:.4f},{angle_penalty:.4f},"
                        f"{lateral_penalty:.4f},{total_cost:.4f}\n")

        return total_cost

    def _compute_radius(self, p1, p2, p3):
        """Estimate curvature radius from 3 points using geometric method."""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        ab = b - a
        bc = c - b

        ab_len = np.linalg.norm(ab)
        bc_len = np.linalg.norm(bc)
        if ab_len < 1e-5 or bc_len < 1e-5:
            return float('inf')

        angle = np.arccos(np.clip(np.dot(ab, bc) / (ab_len * bc_len), -1, 1))
        if angle == 0:
            return float('inf')

        return ab_len / (2 * math.sin(angle / 2))

    def _compute_radius_extended(self, prev2, prev1, curr, next1):
        """Average curvature from two triplets for smoother radius estimation."""
        r1 = self._compute_radius(prev2, prev1, curr)
        r2 = self._compute_radius(prev1, curr, next1)
        if r1 == float('inf') or r2 == float('inf'):
            return float('inf')
        return (r1 + r2) / 2.0

    def reconstruct_path(self, end_node):
        """Rebuild path from goal node to start using parent links."""
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]

    def plan(self, start_world, goal_world, loop_threshold=100):
        """
        A* path planning loop.
        """
        start_x, start_y = self.grid_builder.world_to_grid(*start_world)
        start = Node(start_x, start_y, g=0)

        goal_x, goal_y = self.grid_builder.world_to_grid(*goal_world)
        goal = Node(goal_x, goal_y)

        open_set = []
        heapq.heappush(open_set, (start.f, start))
        visited = set()
        node_map = {(start.x, start.y): start}

        max_iterations = 20000

        while open_set and max_iterations > 0:
            _, current = heapq.heappop(open_set)
            self.nodes_expanded += 1
            visited.add((current.x, current.y))

            if (current.x, current.y) == (goal.x, goal.y):
                return self.reconstruct_path(current)

            for nx, ny in self.get_neighbors(current):
                if (nx, ny) in visited:
                    continue

                neighbor = node_map.get((nx, ny)) or Node(nx, ny)
                if (nx, ny) not in node_map:
                    neighbor.h = self.heuristic(neighbor, goal)
                    node_map[(nx, ny)] = neighbor

                tentative_g = current.g + self.cost(current, neighbor)

                if tentative_g < neighbor.g:
                    neighbor.g = tentative_g
                    neighbor.parent = current
                    heapq.heappush(open_set, (neighbor.f, neighbor))

            max_iterations -= 1

        print("A* failed to reach goal.")
        return None
