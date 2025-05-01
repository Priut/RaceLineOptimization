import numpy as np
from scipy.spatial import ConvexHull, KDTree
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString

class Track:
    """
    Generates smooth, closed racing tracks with spline interpolation and randomness.
    Includes self-intersection checks and curvature constraints to ensure drivability.
    """

    def __init__(self, num_initial_points=20, displacement_scale=15, distance_threshold=5, track_width=20):
        """
        Initializes track generation parameters.
        - num_initial_points: number of seed points for the track
        - displacement_scale: scale of random displacement for variation
        - distance_threshold: minimum spacing between refined points
        - track_width: width of the generated track (used for car constraints)
        """
        self.num_initial_points = num_initial_points
        self.displacement_scale = displacement_scale
        self.distance_threshold = distance_threshold
        self.track_width = track_width

    @staticmethod
    def generate_random_points(num_points, scale=100):
        """Generates uniformly random 2D points within a square of given scale."""
        return np.random.rand(num_points, 2) * scale

    @staticmethod
    def compute_convex_hull(points):
        """Returns the points forming the convex hull of a point cloud (in order)."""
        hull = ConvexHull(points)
        return points[hull.vertices]

    @staticmethod
    def displace_midpoints(points, displacement_scale=10, min_distance=15, max_attempts=10):
        """
        Generates new points by displacing midpoints between pairs of hull points.
        Ensures resulting points are spaced far enough to avoid overlap.
        """
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
                if not new_points:
                    break
                tree = KDTree(new_points)
                if tree.query(new_point, k=1)[0] >= min_distance:
                    break
                # Retry with reduced randomness
                displacement *= 0.6
                new_point = midpoint + displacement + np.random.randn(2) * 0.5
                attempts += 1

            new_points.append(new_point)

        return np.array(new_points)

    @staticmethod
    def refine_points(points, distance_threshold=5):
        """
        Removes points that are too close to each other to avoid jittery splines.
        Keeps only those spaced at least `distance_threshold` apart.
        """
        refined_points = [points[0]]
        for i in range(1, len(points)):
            if np.linalg.norm(points[i] - refined_points[-1]) > distance_threshold:
                refined_points.append(points[i])
        return np.array(refined_points)

    @staticmethod
    def create_spline(points, smoothing=0):
        """
        Interpolates a smooth closed spline through a set of points.
        Returns sampled points along the spline.
        """
        tck, u = splprep(points.T, s=smoothing, per=True)
        u_fine = np.linspace(0, 1, 1000)
        x_fine, y_fine = splev(u_fine, tck)
        return np.array(x_fine), np.array(y_fine)

    @staticmethod
    def check_self_intersection(x, y):
        """
        Checks whether the generated spline intersects itself.
        """
        coords = list(zip(x, y))
        line = LineString(coords)
        return not line.is_simple

    @staticmethod
    def is_curve_too_tight(x, y, min_radius=10):
        """
        Checks whether any part of the track has curvature too tight (below `min_radius`).
        Prevents creating sharp corners that are undrivable.
        """
        if len(x) < 3:
            return False

        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2 + 1e-5) ** 1.5
        with np.errstate(divide='ignore'):
            radius = 1.0 / (curvature + 1e-8)

        return np.any(radius < min_radius)

    def generate_map(self):
        """
        Generates a closed track that is:
        - Non-intersecting
        - Smooth and round
        - With no tight turns

        Returns:
            - x_spline, y_spline: interpolated centerline coordinates
            - track_width: constant track width for simulation
        """
        for attempt in range(10):
            points = self.generate_random_points(self.num_initial_points)
            hull_points = self.compute_convex_hull(points)
            displaced = self.displace_midpoints(hull_points, self.displacement_scale)

            x_spline, y_spline = self.create_spline(displaced)

            if not self.check_self_intersection(x_spline, y_spline) and \
               not self.is_curve_too_tight(x_spline, y_spline, min_radius=2):
                return x_spline, y_spline, self.track_width

        print("Failed to generate a valid track after 10 attempts.")
        return x_spline, y_spline, self.track_width
