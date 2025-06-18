import numpy as np
from scipy.interpolate import splprep, splev

class Car:
    """
    Simulates a vehicle driving along a predefined spline path.
    Includes physics-based acceleration, braking, drag, and curvature-based speed adaptation.
    """

    def __init__(self, x_spline, y_spline, track_width):
        # Track spline and geometry
        self.car_x = x_spline
        self.car_y = y_spline
        self.track_width = track_width
        self.path_points = np.column_stack((x_spline, y_spline))

        # Vehicle state
        self.speed = 2.0  # initial speed (m/s)
        self.throttle = 0.5
        self.acceleration = 0.0
        self.max_speed_reached = self.speed
        self.distance_traveled = 0.0
        self.car_index = 0
        self.steering_angle = 0.0

        # Vehicle parameters
        self.mass = 1350  # kg
        self.engine_power = 275000  # W
        self.transmission_efficiency = 0.9
        self.Cd = 0.35  # drag coefficient
        self.rho = 1.225  # air density
        self.A = 2.0  # frontal area (m^2)
        self.Cr = 0.01  # rolling resistance coefficient
        self.g = 9.81  # gravity

        # Precompute path distances and total length
        self.distances = np.sqrt(np.sum(np.diff(self.path_points, axis=0) ** 2, axis=1))
        self.cumulative_distance = np.insert(np.cumsum(self.distances), 0, 0)
        self.total_length = self.cumulative_distance[-1]

        # Spline for curvature analysis
        tck, _ = splprep([self.car_x, self.car_y], u=self.cumulative_distance, s=0)
        self.curvature_spline = tck

    def compute_acceleration(self):
        """
        Computes longitudinal acceleration based on throttle, drag, and rolling resistance.
        """
        if self.throttle < 0:  # Braking
            brake_force = abs(self.throttle) * self.mass * self.g * 0.6
            self.acceleration = -brake_force / self.mass
        else:  # Accelerating
            effective_speed = max(self.speed, 1.0)
            engine_force = (self.engine_power * self.transmission_efficiency) / effective_speed
            drag_force = 0.5 * self.Cd * self.rho * self.A * self.speed ** 2
            rolling_force = self.Cr * self.mass * self.g

            net_force = engine_force * self.throttle - drag_force - rolling_force
            self.acceleration = net_force / self.mass

        return self.acceleration

    def compute_local_curvature_radius_by_distance(self):
        """
        Returns the local curvature radius at the current distance along the path.
        """
        u = self.distance_traveled
        dx, dy = splev(u, self.curvature_spline, der=1)
        ddx, ddy = splev(u, self.curvature_spline, der=2)

        numerator = abs(dx * ddy - dy * ddx)
        denominator = (dx ** 2 + dy ** 2) ** 1.5
        if denominator == 0:
            return float('inf')

        curvature = numerator / denominator
        return 1 / curvature if curvature != 0 else float('inf')

    def compute_max_lateral_speed(self, radius, mu=1.3):
        """
        Computes the maximum lateral speed based on curvature radius and friction coefficient.
        """
        if radius == float('inf'):
            return float('inf')
        return np.sqrt(mu * self.g * radius)

    def move_car(self, dt=0.016):
        """
        Advances the car forward by dt seconds using basic longitudinal dynamics.
        Adjusts throttle based on curvature-constrained safe speed.
        """
        radius = self.compute_local_curvature_radius_by_distance()

        # Smooth curvature estimate for throttle control
        self.smoothed_radius = 0.9 * getattr(self, "smoothed_radius", radius) + 0.1 * radius
        safe_speed = self.compute_max_lateral_speed(self.smoothed_radius)

        # Ratio of current speed to curvature-constrained limit
        speed_ratio = self.speed / (safe_speed + 1e-5)

        # Map speed ratio to throttle target (linear response curve)
        target_throttle = 1.5 - 2.0 * speed_ratio
        target_throttle = np.clip(target_throttle, -1.0, 1.0)

        # Smooth throttle transition
        throttle_response = 2.0
        if self.throttle < target_throttle:
            self.throttle += throttle_response * dt
        elif self.throttle > target_throttle:
            self.throttle -= throttle_response * dt
        self.throttle = np.clip(self.throttle, -1, 1)

        # Update physics
        self.compute_acceleration()
        self.speed += self.acceleration * dt
        self.speed = max(self.speed, 0)
        self.max_speed_reached = max(self.max_speed_reached, self.speed)

        # Advance along path
        self.distance_traveled += self.speed * dt
        self.distance_traveled = min(self.distance_traveled, self.total_length)

    def get_position(self):
        """
        Returns the current position of the car on the path, or None if finished.
        """
        if self.is_finished():
            return None
        return splev(self.distance_traveled, self.curvature_spline)

    def is_finished(self):
        """
        Returns True if the car reached the end of the path.
        """
        return self.distance_traveled >= self.total_length
