import numpy as np

class Car:
    def __init__(self, x_spline, y_spline, track_width):
        self.car_x = x_spline
        self.car_y = y_spline
        self.track_width = track_width
        self.car_index = 0
        self.speed = 2  # Start with a small initial speed (m/s)
        self.acceleration = 0  # m/s²
        self.throttle = 0.5  # Start with half throttle (can be adjusted)
        self.mass = 1350  # kg (GT3 car)
        self.engine_power = 275000  # W (~500 HP)
        self.transmission_efficiency = 0.9

        # Drag force parameters
        self.Cd = 0.35
        self.rho = 1.225
        self.A = 2.0

        # Rolling resistance parameters
        self.Cr = 0.01
        self.g = 9.81

        self.speed_profile = self.compute_speed_profile(self.car_x, self.car_y)

    def compute_acceleration(self):
        self.throttle = np.clip(self.throttle, 0, 1)
        min_speed = 1
        effective_speed = max(self.speed, min_speed)
        force_accel = (self.engine_power * self.transmission_efficiency) / effective_speed
        accel_throttle = (force_accel * self.throttle) / self.mass
        force_drag = 0.5 * self.Cd * self.rho * self.A * (self.speed ** 2)
        accel_drag = force_drag / self.mass
        force_rolling = self.Cr * self.mass * self.g
        accel_rolling = force_rolling / self.mass
        self.acceleration = accel_throttle - accel_drag - accel_rolling
        return self.acceleration

    def compute_speed_profile(self, track_x, track_y, base_speed=2.0):
        dx = np.gradient(track_x)
        dy = np.gradient(track_y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-5) ** 1.5
        max_curvature = np.max(curvature)
        speed = base_speed * (1 - curvature / (max_curvature + 1e-5))
        speed = np.clip(speed, 0.5, base_speed)
        return speed

    def compute_local_curvature_radius(self, index):
        if index <= 0 or index >= len(self.car_x) - 1:
            return float('inf')

        dx = (self.car_x[index + 1] - self.car_x[index - 1]) / 2
        dy = (self.car_y[index + 1] - self.car_y[index - 1]) / 2
        ddx = self.car_x[index + 1] - 2 * self.car_x[index] + self.car_x[index - 1]
        ddy = self.car_y[index + 1] - 2 * self.car_y[index] + self.car_y[index - 1]

        numerator = abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2) ** 1.5

        if denominator == 0:
            return float('inf')

        curvature = numerator / denominator
        radius = 1 / curvature if curvature != 0 else float('inf')
        return radius

    def compute_max_lateral_speed(self, radius, mu=1.3):
        if radius == float('inf'):
            return float('inf')
        return np.sqrt(mu * self.g * radius)

    def move_car(self, dt=0.016):
        """
        Updates car's speed and position with smooth throttle/brake transitions based on curve radius.
        """
        # --- Curve-based logic ---
        radius = self.compute_local_curvature_radius(self.car_index)
        safe_speed = self.compute_max_lateral_speed(radius)

        # Determine target throttle (range: 0 = braking, 1 = full throttle)
        if self.speed > safe_speed:
            target_throttle = 0.2  # reduce throttle
        else:
            target_throttle = 0.6  # accelerate

        # --- Smooth throttle transition ---
        throttle_response = 1.5  # units per second (can be tuned)
        if self.throttle < target_throttle:
            self.throttle += throttle_response * dt
        elif self.throttle > target_throttle:
            self.throttle -= throttle_response * dt

        self.throttle = np.clip(self.throttle, 0, 1)

        # --- Continue physics ---
        self.compute_acceleration()
        self.speed += self.acceleration * dt
        self.speed = max(self.speed, 0)

        # Move forward
        if self.car_index < len(self.car_x) - 1:
            self.car_index += int(self.speed * dt * 10)
            self.car_index = min(self.car_index, len(self.car_x) - 1)

    def get_position(self):
        if self.car_index < len(self.car_x):
            return self.car_x[self.car_index], self.car_y[self.car_index]
        return None
