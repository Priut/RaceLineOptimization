import numpy as np
from scipy.spatial import ConvexHull, KDTree
from scipy.interpolate import splprep, splev

class Track:
    def __init__(self, num_initial_points=20, displacement_scale=15, distance_threshold=5, track_width=20):
        self.num_initial_points = num_initial_points
        self.displacement_scale = displacement_scale
        self.distance_threshold = distance_threshold
        self.track_width = track_width

    @staticmethod
    def generate_random_points(num_points, scale=100):
        return np.random.rand(num_points, 2) * scale

    @staticmethod
    def compute_convex_hull(points):
        hull = ConvexHull(points)
        return points[hull.vertices]

    @staticmethod
    def displace_midpoints(points, displacement_scale=10, min_distance=15, max_attempts=10):
        """ Prevents overlapping by ensuring no two points are too close. """
        new_points = []
        num_points = len(points)

        for i in range(num_points):
            p1 = points[i]
            p2 = points[(i + 1) % num_points]
            midpoint = (p1 + p2) / 2
            displacement = np.random.randn(2) * displacement_scale

            new_point = midpoint + displacement
            attempts = 0

            while attempts < max_attempts:
                tree = KDTree(new_points) if new_points else None
                if tree is None or tree.query(new_point, k=1)[0] >= min_distance:
                    break  # Valid point found, exit loop
                displacement *= 0.6  # Reduce displacement faster
                new_point = midpoint + displacement + np.random.randn(2) * 0.5  # Small perturbation
                attempts += 1

            new_points.append(new_point)

        return np.array(new_points)

    @staticmethod
    def remove_self_intersections(points, min_distance=10):
        """ Detects and removes points that cause self-intersections. """
        tree = KDTree(points)
        filtered_points = [points[0]]

        for i in range(1, len(points)):
            if tree.query(points[i], k=2)[0][1] > min_distance:
                filtered_points.append(points[i])

        return np.array(filtered_points)

    @staticmethod
    def refine_points(points, distance_threshold=5):
        """ Ensures a smooth track by keeping points evenly spaced. """
        refined_points = [points[0]]
        for i in range(1, len(points)):
            if np.linalg.norm(points[i] - refined_points[-1]) > distance_threshold:
                refined_points.append(points[i])
        return np.array(refined_points)

    @staticmethod
    def create_spline(points, smoothing=0):
        """ Generates a smooth spline through the track points. """
        tck, u = splprep(points.T, s=smoothing, per=True)
        u_fine = np.linspace(0, 1, 1000)
        x_fine, y_fine = splev(u_fine, tck)
        return np.array(x_fine), np.array(y_fine)

    def generate_map(self):
        """Generates the entire racing track with boundaries and returns track data"""
        points = self.generate_random_points(self.num_initial_points)
        hull_points = self.compute_convex_hull(points)
        displaced_points = self.displace_midpoints(hull_points, self.displacement_scale)
        non_intersecting_points = self.remove_self_intersections(displaced_points)
        refined_points = self.refine_points(non_intersecting_points, self.distance_threshold)
        x_spline, y_spline = self.create_spline(refined_points)

        return x_spline, y_spline, self.track_width
