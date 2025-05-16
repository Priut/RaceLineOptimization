import numpy as np
from model.car import Car
import os

class RacingEnv:
    def __init__(self, x_spline, y_spline, track_width, dt=0.1, max_steer_change=3.0):
        """
        Initializes the racing environment with the given track centerline and width.

        Parameters:
            x_spline (np.ndarray): X coordinates of the centerline.
            y_spline (np.ndarray): Y coordinates of the centerline.
            track_width (float): Width of the track.
            dt (float): Time step for simulation.
            max_steer_change (float): Maximum change in steering angle per step.
        """
        # Track Geometry 
        self.center_x = x_spline
        self.center_y = y_spline
        self.track_width = track_width
        self.dt = dt
        self.max_steer_change = max_steer_change

        # Derived Track Properties
        self.arc_lengths = self._compute_arc_lengths(self.center_x, self.center_y)
        self.track_length = len(self.center_x)
        self.total_track_distance = self.arc_lengths[-1]

        # Track Behavior Analysis 
        self.curvatures = self._precompute_curvatures()
        self.curve_segments = self._detect_curve_segments()

        # Logging Setup 
        self.logging_enabled = False
        self.log_file = "reward_log.txt"
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        # Initial State 
        self.reset()

    def reset(self):
        """
        Resets the environment state for a new episode.

        Returns:
            tuple: Initial state vector.
        """
        self.car = Car(self.center_x, self.center_y, self.track_width)
        self.path_x = []
        self.path_y = []

        self.last_action = 0.0
        self.offset = 0.0
        self.last_offset = 0.0
        self.position = 0

        return self.get_state()

    def step(self, action):
        """
        Performs one environment step:
        - Applies the action (steering adjustment),
        - Moves the car and updates offset,
        - Calculates reward,
        - Updates internal state and returns next state, reward, and done flag.

        Parameters:
            action (float): Steering adjustment input.

        Returns:
            tuple: (next_state, reward, done)
        """

        # 1. Apply Steering Action 
        delta = action * self.max_steer_change
        self.car.steering_angle = np.clip(self.car.steering_angle + delta, -0.5, 0.5)

        # Move car forward
        self.car.move_car(self.dt)

        # 2. Update Offset 
        lateral_speed = self.car.speed * np.tan(self.car.steering_angle)
        lateral_movement = 0.2 * lateral_speed * self.dt
        self.offset = np.clip(self.offset + lateral_movement,
                              -self.track_width / 2.2,
                              self.track_width / 2.2)

        # 3. Check if Lap is Complete
        done = self.car.distance_traveled >= 0.999 * self.total_track_distance

        # 4. Update Car Position Along Track 
        pos = self.car.get_position()
        if pos is not None:
            x, y = pos
            position_idx = self._find_closest_index(self.car.distance_traveled)
            self.position = position_idx

            # Compute direction of track and its normal
            center_x = self.center_x[position_idx]
            center_y = self.center_y[position_idx]

            next_idx = min(position_idx + 1, self.track_length - 1)
            direction = np.array([
                self.center_x[next_idx] - center_x,
                self.center_y[next_idx] - center_y
            ])
            normal = np.array([-direction[1], direction[0]])
            normal /= (np.linalg.norm(normal) + 1e-8)

            # Apply lateral offset to car position
            x += self.offset * normal[0]
            y += self.offset * normal[1]

            self.path_x.append(x)
            self.path_y.append(y)

        # 5. Calculate Reward 
        reward = self._compute_reward(action)

        # 6. Get Next State 
        next_state = self.get_state()

        # 7. Update Last Offset and Action 
        self.last_offset = self.offset
        self.last_action = action

        return next_state, reward, done

    def get_state(self):
        """
        Returns a 7-dimensional state vector used for learning:
        - Normalized position along the track,
        - Normalized lateral offset,
        - Offset change rate,
        - Steering angle,
        - Curvature at current and upcoming points (5 and 10 units ahead).

        Returns:
            tuple: state vector
        """
        normalized_pos = self.car.distance_traveled / self.total_track_distance
        norm_offset = self.offset / (self.track_width / 2.2)
        offset_velocity = (self.offset - self.last_offset) / self.dt
        steering_angle = self.car.steering_angle

        curvature_now = self.curvatures[self.position]
        curvature_5 = self.curvatures[self._index_at_distance(self.position, 5)]
        curvature_10 = self.curvatures[self._index_at_distance(self.position, 10)]

        return (
            normalized_pos,
            norm_offset,
            offset_velocity,
            steering_angle,
            curvature_now,
            curvature_5,
            curvature_10
        )

    def _compute_reward(self, action):
        """
        Computes the total reward for the current state, based on:
        - vehicle progress along the track,
        - alignment with the ideal offset from the center line (depending on track behavior),
        - driving smoothness,
        - boundary violations.

        Parameters:
            action (float): Steering action taken by the agent.

        Returns:
            float: Total reward score for the step.
        """

        # Base progress reward (how much forward motion was made) 
        progress_reward = self.car.speed * self.dt

        # Normalize current offset 
        norm_offset = self.offset / (self.track_width / 2.2)
        behavior = self.point_behaviors[self.position]

        # Determine ideal track offset based on behavior 
        desired_offset = 0.0  # default for straight

        if behavior == "straight":
            pass
        else:
            stage, direction = behavior
            curvature_strength = abs(self.curvatures[self.position])
            curve_importance = np.clip(curvature_strength * 2000, 0.0, 1.0)

            match stage:
                case "adjust":
                    # Look ahead for the next real curve stage
                    next_behavior = None
                    lookahead = self.position + 1
                    while lookahead < len(self.point_behaviors):
                        next_b = self.point_behaviors[lookahead]
                        if next_b != "straight" and next_b[0] != "adjust":
                            next_behavior = next_b
                            break
                        lookahead += 1

                    if next_behavior and next_behavior[0] == "entry":
                        desired_offset = 1.0 * direction  # Prepare for turn
                    else:
                        desired_offset = 0.0  # Stay centered otherwise

                case "entry":
                    desired_offset = (0.8 - 0.4 * curve_importance) * (-direction)

                case "apex":
                    desired_offset = -1.1 * direction  # Hug the inner curve

                case "exit":
                    desired_offset = (0.8 - 0.3 * curve_importance) * direction

        # Compute offset error
        offset_error = abs(norm_offset - desired_offset)

        # Reward offset alignment depending on stage 
        if behavior == "straight":
            offset_weight = 8
            offset_reward = offset_weight * (1.2 - (offset_error * 2.0) ** 2)
        else:
            match stage:
                case "apex":
                    offset_weight = 6.0
                case "entry" | "exit":
                    offset_weight = 3.0
                case _:
                    offset_weight = 2.0
            offset_reward = offset_weight * (1.5 - (offset_error * 3.0) ** 2)

        offset_reward = max(offset_reward, -5.0)

        # Bonus for excellent apex hit
        if behavior != "straight" and stage == "apex" and offset_error < 0.1:
            offset_reward += 2.0

        # Bonus for crossing over the ideal offset (indicates movement toward it)
        if (norm_offset - desired_offset) * (self.last_offset - desired_offset) < 0:
            offset_reward += 1.0 if (behavior != "straight" and stage == "apex") else 0.5

        # Progress multiplier if on the right line
        if behavior != "straight" and offset_error < 0.2:
            progress_reward *= 1.8 if stage == "apex" else 2.0
        elif behavior == "straight" and offset_error < 0.1:
            progress_reward *= 1.2

        # Penalties 
        steering_penalty = -0.05 * abs(action)
        smoothness_penalty = -0.2 * abs(action - self.last_action)
        boundary_penalty = -2.0 if abs(self.offset) > (self.track_width / 2.2) * 0.95 else 0.0

        # Final reward composition
        total_reward = (
                progress_reward +
                offset_reward +
                steering_penalty +
                smoothness_penalty +
                boundary_penalty
        )

        # Optional logging 
        if self.logging_enabled:
            with open(self.log_file, "a") as f:
                f.write(
                    f"pos={self.position}, stage={behavior}, norm_offset={norm_offset:.2f}, "
                    f"desired={desired_offset:.2f}, offset_error={offset_error:.2f}, "
                    f"offset_reward={offset_reward:.2f}, progress_reward={progress_reward:.2f}, "
                    f"steering_penalty={steering_penalty:.2f}, smoothness_penalty={smoothness_penalty:.2f}, "
                    f"boundary_penalty={boundary_penalty:.2f}, total={total_reward:.2f}\n"
                )

        return total_reward

    def _precompute_curvatures(self):
        """
        Computes a smoothed curvature profile along the track.

        Returns:
            list: Smoothed curvature values at each track point.
        """
        raw_curvs = [self._compute_curvature(i) for i in range(self.track_length)]

        # Smooth curvature values using a simple moving average
        window = 5
        smooth_curvs = np.convolve(raw_curvs, np.ones(window) / window, mode='same')

        return smooth_curvs.tolist()

    def _detect_curve_segments(self, threshold=0.003):
        """
        Detects curved segments based on curvature threshold and labels point behaviors.

        Parameters:
            threshold (float): Minimum absolute curvature to be considered part of a curve.

        Returns:
            list: List of (start_idx, end_idx, direction) for each detected curve segment.
        """
        segments = []
        point_behaviors = ["straight"] * len(self.curvatures)

        in_curve = False
        start_idx = 0
        direction = 0
        extend_after = 10
        extend_counter = 0

        # Detect curved segments and mark them 
        for i, curv in enumerate(self.curvatures):
            if abs(curv) > threshold:
                if not in_curve:
                    in_curve = True
                    start_idx = i  # 🔥 precise start of curve
                    direction = -1 if curv > 0 else 1
                extend_counter = extend_after  # Reset extension delay
            else:
                if in_curve:
                    if extend_counter > 0:
                        extend_counter -= 1
                    else:
                        end_idx = i - 1
                        self._label_curve(point_behaviors, start_idx, end_idx, direction)
                        segments.append((start_idx, end_idx, direction))
                        in_curve = False

        # Handle curve that reaches the end
        if in_curve:
            end_idx = len(self.curvatures) - 1
            self._label_curve(point_behaviors, start_idx, end_idx, direction)
            segments.append((start_idx, end_idx, direction))

        # Detect if the curve wraps around from end to start
        if point_behaviors[0] != "straight" and point_behaviors[-1] != "straight":
            # Find the contiguous curve at the start
            start_idx = 0
            while start_idx < len(point_behaviors) and point_behaviors[start_idx] != "straight":
                start_idx += 1

            # Find the contiguous curve at the end
            end_idx = len(point_behaviors) - 1
            while end_idx >= 0 and point_behaviors[end_idx] != "straight":
                end_idx -= 1

            # Relabel this wrap-around curve as straight
            for i in list(range(0, start_idx)) + list(range(end_idx + 1, len(point_behaviors))):
                point_behaviors[i] = "straight"

        self.point_behaviors = point_behaviors
        return segments

    def _label_curve(self, behaviors, start_idx, end_idx, direction):
        """
        Labels a curved segment with appropriate driving stages:
        entry, apex, exit, and surrounding adjust zones.

        Parameters:
            behaviors (list): List of point behaviors to update.
            start_idx (int): Start index of the curve.
            end_idx (int): End index of the curve.
            direction (int): Turn direction (-1 = left, 1 = right).
        """
        curve_len = end_idx - start_idx + 1

        if curve_len <= 5:
            # Short curves are labeled entirely as apex
            for j in range(start_idx, end_idx + 1):
                behaviors[j] = ("apex", direction)
            return

        # Find point with max curvature for apex center
        apex_idx = start_idx + np.argmax(np.abs(self.curvatures[start_idx:end_idx + 1]))

        # Apex region size (30% of curve length)
        apex_zone_size = max(3, int(curve_len * 0.3))
        apex_half = apex_zone_size // 2
        apex_start = max(start_idx, apex_idx - apex_half)
        apex_end = min(end_idx, apex_idx + apex_half)

        # Entry/exit lengths
        before_apex_len = apex_start - start_idx
        after_apex_len = end_idx - apex_end

        # Determine adjust zones before and after the curve
        adjust_len_entry = int(before_apex_len / 2)
        adjust_len_exit = int(after_apex_len / 2)

        adjust_start_entry = max(0, start_idx - adjust_len_entry)
        adjust_end_entry = start_idx - 1
        adjust_start_exit = end_idx + 1
        adjust_end_exit = min(self.track_length - 1, end_idx + adjust_len_exit)

        # --- Label pre-curve adjust/entry zone ---
        if adjust_end_entry >= adjust_start_entry:
            for j in range(adjust_start_entry, adjust_end_entry + 1):
                if behaviors[j] != "straight":
                    mid = (adjust_start_entry + adjust_end_entry) // 2
                    behaviors[j] = ("adjust", direction) if j > mid else ("entry", direction)
                else:
                    behaviors[j] = ("adjust", direction)

        # --- Label post-curve exit/adjust zone ---
        if adjust_end_exit >= adjust_start_exit:
            for j in range(adjust_start_exit, adjust_end_exit + 1):
                if behaviors[j] != "straight":
                    mid = (adjust_start_exit + adjust_end_exit) // 2
                    behaviors[j] = ("adjust", direction) if j > mid else ("exit", direction)
                else:
                    behaviors[j] = ("adjust", direction)

        # --- Label the curve: entry → apex → exit ---
        for j in range(start_idx, end_idx + 1):
            if start_idx <= j < apex_start:
                behaviors[j] = ("entry", direction)
            elif apex_start <= j <= apex_end:
                behaviors[j] = ("apex", direction)
            elif apex_end < j <= end_idx:
                behaviors[j] = ("exit", direction)

    def _find_closest_index(self, distance):
        """
        Returns the index of the point closest to the given arc length distance.

        Parameters:
            distance (float): Distance along the track.

        Returns:
            int: Closest index in arc_lengths array.
        """
        idx = np.searchsorted(self.arc_lengths, distance)
        return np.clip(idx, 0, self.track_length - 1)

    def _index_at_distance(self, start_idx, distance):
        """
        Returns the index of the point that lies a certain distance ahead from start_idx.

        Parameters:
            start_idx (int): Starting index.
            distance (float): Distance to travel forward along arc length.

        Returns:
            int: Index of point at or beyond the specified distance.
        """
        target_distance = self.arc_lengths[start_idx] + distance
        for i in range(start_idx, len(self.arc_lengths)):
            if self.arc_lengths[i] >= target_distance:
                return i
        return len(self.arc_lengths) - 1

    def _compute_arc_lengths(self, x, y):
        """
        Computes cumulative arc length along the spline path.

        Parameters:
            x (ndarray): x coordinates of the path.
            y (ndarray): y coordinates of the path.

        Returns:
            ndarray: Array of cumulative distances at each point.
        """
        distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        return np.concatenate(([0], np.cumsum(distances)))

    def _compute_curvature(self, idx):
        """
        Estimates the curvature at a given index using finite differences.

        Parameters:
            idx (int): Index of the point.

        Returns:
            float: Curvature value (0.0 at endpoints).
        """
        if idx <= 0 or idx >= self.track_length - 1:
            return 0.0

        dx = (self.center_x[idx + 1] - self.center_x[idx - 1]) / 2
        dy = (self.center_y[idx + 1] - self.center_y[idx - 1]) / 2
        ddx = self.center_x[idx + 1] - 2 * self.center_x[idx] + self.center_x[idx - 1]
        ddy = self.center_y[idx + 1] - 2 * self.center_y[idx] + self.center_y[idx - 1]

        num = abs(dx * ddy - dy * ddx)
        denom = (dx ** 2 + dy ** 2 + 1e-8) ** 1.5
        return num / denom

