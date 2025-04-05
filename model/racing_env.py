import numpy as np

class RacingEnv:
    def __init__(self, x_spline, y_spline, track_width, max_steer_change=0.05, dt=0.1):
        self.center_x = x_spline
        self.center_y = y_spline
        self.track_width = track_width
        self.dt = dt  # time per step
        self.max_offset = track_width / 2.2  # max lateral from center
        self.max_steer_change = max_steer_change  # steering delta per step
        self.actions = [-1, 0, 1]  # steer left, stay, right
        self.track_length = len(x_spline)
        self.arc_lengths = self._compute_arc_lengths(self.center_x, self.center_y)
        self.reset()

    def reset(self):
        self.position = 0
        self.steering_angle = 0.0
        self.offset = 0.0
        self.path_x = []
        self.path_y = []
        return self.get_state()

    def get_state(self):
        """
        Returns a richer state representation:
        - normalized position along track
        - lateral offset (normalized)
        - offset velocity (delta offset)
        - current steering angle
        - local curvature (now, +5, +10)
        """
        normalized_pos = self.position / self.track_length
        norm_offset = self.offset / self.max_offset

        # Estimate offset velocity
        if hasattr(self, 'last_offset'):
            offset_velocity = (self.offset - self.last_offset) / self.dt
        else:
            offset_velocity = 0.0
        self.last_offset = self.offset

        # Local + future curvature
        curvature_now = self._compute_curvature(self.position)
        look_5m = self._index_at_distance(self.position, 5)
        look_10m = self._index_at_distance(self.position, 10)

        curvature_5 = self._compute_curvature(look_5m)
        curvature_10 = self._compute_curvature(look_10m)

        return (
            normalized_pos,
            norm_offset,
            offset_velocity,
            self.steering_angle,
            curvature_now,
            curvature_5,
            curvature_10
        )

    def step(self, action):
        # Action changes steering angle
        delta = action * self.max_steer_change
        self.steering_angle += delta
        self.steering_angle = np.clip(self.steering_angle, -0.5, 0.5)  # radians

        # Integrate lateral offset using steering
        velocity = 10.0  # constant forward velocity
        lateral_change = velocity * np.sin(self.steering_angle) * self.dt
        self.offset += lateral_change
        self.offset = np.clip(self.offset, -self.max_offset, self.max_offset)

        # Get normal direction to compute world position
        dx = np.gradient(self.center_x)
        dy = np.gradient(self.center_y)
        norm = np.sqrt(dx[self.position] ** 2 + dy[self.position] ** 2)
        if norm == 0:
            norm = 1
        dx /= norm
        dy /= norm
        normal_x = dy[self.position]
        normal_y = -dx[self.position]

        x = self.center_x[self.position] + self.offset * normal_x
        y = self.center_y[self.position] + self.offset * normal_y
        self.path_x.append(x)
        self.path_y.append(y)

        self.position += 1
        done = self.position >= self.track_length - 1

        # Compute curvature and steering cost
        curvature = self._compute_curvature(self.position)
        offset_penalty = abs(self.offset)

        # --- Reward Shaping ---
        progress_reward = 1.0  # reward for moving forward

        # Steering penalty
        if hasattr(self, 'last_action'):
            steering_penalty = abs(action - self.last_action) * 0.1
        else:
            steering_penalty = 0.0
        self.last_action = action

        # Adaptive offset penalty (harder in turns)
        curvature_weight = np.clip(curvature * 10, 0.1, 1.0)
        adaptive_offset_penalty = curvature_weight * offset_penalty

        # Offset jump penalty
        if hasattr(self, 'last_offset'):
            offset_jump_penalty = abs(self.offset - self.last_offset) * 0.2
        else:
            offset_jump_penalty = 0.0
        self.last_offset = self.offset

        # Center bonus (only on straights)
        if curvature < 0.02:
            center_bonus = max(0, 1 - abs(self.offset) / self.max_offset) * 0.5
        else:
            center_bonus = 0

        # Final reward
        reward = (
                + 1.0 / (1.0 + curvature)
                - 0.02 * adaptive_offset_penalty
                - steering_penalty
                - offset_jump_penalty
                + progress_reward
                + center_bonus
        )

        return self.get_state(), reward, done

    def _compute_curvature(self, idx):
        if idx <= 0 or idx >= self.track_length - 1:
            return 0.0

        dx = (self.center_x[idx + 1] - self.center_x[idx - 1]) / 2
        dy = (self.center_y[idx + 1] - self.center_y[idx - 1]) / 2
        ddx = self.center_x[idx + 1] - 2 * self.center_x[idx] + self.center_x[idx - 1]
        ddy = self.center_y[idx + 1] - 2 * self.center_y[idx] + self.center_y[idx - 1]

        num = abs(dx * ddy - dy * ddx)
        denom = (dx ** 2 + dy ** 2 + 1e-5) ** 1.5
        return num / denom if denom != 0 else 0.0

    def _compute_arc_lengths(self, x, y):
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        arc_lengths = np.concatenate(([0], np.cumsum(distances)))
        return arc_lengths

    def _index_at_distance(self, start_idx, distance):
        target_distance = self.arc_lengths[start_idx] + distance
        for i in range(start_idx, len(self.arc_lengths)):
            if self.arc_lengths[i] >= target_distance:
                return i
        return len(self.arc_lengths) - 1  # fallback to last point

