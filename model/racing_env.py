import numpy as np
from model.car import Car
import os

class RacingEnv:
    def __init__(self, x_spline, y_spline, track_width, dt=0.1, max_steer_change=3.0):
        self.center_x = x_spline
        self.center_y = y_spline
        self.track_width = track_width
        self.dt = dt
        self.max_steer_change = max_steer_change

        self.arc_lengths = self._compute_arc_lengths(self.center_x, self.center_y)
        self.track_length = len(self.center_x)
        self.total_track_distance = self.arc_lengths[-1]

        self.curvatures = self._precompute_curvatures()
        self.curve_segments = self._detect_curve_segments()
        self.logging_enabled = False

        self.log_file = "reward_log.txt"
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        self.reset()

    def reset(self):
        self.car = Car(self.center_x, self.center_y, self.track_width)
        self.path_x = []
        self.path_y = []
        self.last_action = 0.0
        self.offset = 0.0
        self.last_offset = 0.0
        self.position = 0
        return self.get_state()

    def step(self, action):
        # 1. Apply Action
        delta = action * self.max_steer_change
        self.car.steering_angle = np.clip(self.car.steering_angle + delta, -0.5, 0.5)

        self.car.move_car(self.dt)

        # 2. Update Offset
        lateral_speed = self.car.speed * np.tan(self.car.steering_angle)
        lateral_movement = 0.2 * lateral_speed * self.dt
        self.offset = np.clip(self.offset + lateral_movement, -self.track_width / 2.2, self.track_width / 2.2)

        # 3. Determine Done State
        done = self.car.distance_traveled >= 0.999 * self.total_track_distance

        # 4. Update Car Position
        pos = self.car.get_position()
        if pos is not None:
            x, y = pos
            position_idx = self._find_closest_index(self.car.distance_traveled)
            self.position = position_idx

            # Calculate normal vector
            center_x = self.center_x[position_idx]
            center_y = self.center_y[position_idx]
            track_direction = np.array([
                self.center_x[min(position_idx + 1, self.track_length - 1)] - center_x,
                self.center_y[min(position_idx + 1, self.track_length - 1)] - center_y
            ])
            normal = np.array([-track_direction[1], track_direction[0]])
            normal /= (np.linalg.norm(normal) + 1e-8)

            # Apply Offset
            x += self.offset * normal[0]
            y += self.offset * normal[1]

            self.path_x.append(x)
            self.path_y.append(y)

        # 5. Calculate Reward
        reward = self._compute_reward(action)

        # 6. Get Next State
        next_state = self.get_state()

        # 7. Update last variables
        self.last_offset = self.offset
        self.last_action = action

        return next_state, reward, done

    def get_state(self):
        normalized_pos = self.car.distance_traveled / self.total_track_distance
        norm_offset = self.offset / (self.track_width / 2.2)
        offset_velocity = (self.offset - self.last_offset) / self.dt
        steering_angle = self.car.steering_angle

        curvature_now = self.curvatures[self.position]
        curvature_5 = self.curvatures[self._index_at_distance(self.position, 5)]
        curvature_10 = self.curvatures[self._index_at_distance(self.position, 10)]

        return (normalized_pos, norm_offset, offset_velocity, steering_angle,
                curvature_now, curvature_5, curvature_10)

    def _compute_reward(self, action):
        progress_reward = self.car.speed * self.dt
        norm_offset = self.offset / (self.track_width / 2.2)
        behavior = self.point_behaviors[self.position]

        if behavior == "straight":
            desired_offset = 0.0
        else:
            stage, direction = behavior
            curvature_strength = abs(self.curvatures[self.position])
            curve_importance = np.clip(curvature_strength * 2000, 0.0, 1.0)

            if stage == "adjust":
                next_behavior = None
                lookahead = self.position + 1
                while lookahead < len(self.point_behaviors):
                    behavior = self.point_behaviors[lookahead]
                    if behavior != "straight" and behavior[0] != "adjust":
                        next_behavior = behavior
                        break
                    lookahead += 1

                if next_behavior and next_behavior[0] == "entry":
                    desired_offset = 1.0 * direction  # 🔥 Stay wide before a curve
                else:
                    desired_offset = 0.0  # 🏁 Center after exit
            elif stage == "entry":
                desired_offset = (0.8 - 0.4 * curve_importance) * (-direction)  # Deep toward apex
            elif stage == "apex":
                desired_offset = -1.1 * direction  # Inside hard at apex
            elif stage == "exit":
                desired_offset = (0.8 - 0.3 * curve_importance) * direction  # Open wide again

        offset_error = abs(norm_offset - desired_offset)

        # --- NEW: Sharper reward curve ---
        if behavior == "straight":
            offset_weight = 8
            offset_reward = offset_weight * (1.2 - (offset_error * 2.0) ** 2)
        else:
            if stage == "apex":
                offset_weight = 6.0
            elif stage in ["entry", "exit"]:
                offset_weight = 3.0
            else:
                offset_weight = 2.0
            offset_reward = offset_weight * (1.5 - (offset_error * 3.0) ** 2)

        offset_reward = max(offset_reward, -5.0)

        # Bonus for very good apex hit
        if behavior != "straight" and stage == "apex" and offset_error < 0.1:
            offset_reward += 2.0  # Huge boost for perfect apex

        if (norm_offset - desired_offset) * (self.last_offset - desired_offset) < 0:
            offset_reward += 1.0 if (behavior != "straight" and stage == "apex") else 0.5

        # Progress adjustment
        if behavior != "straight" and offset_error < 0.2:
            if stage == "apex":
                progress_reward *= 1.8
            else:
                progress_reward *= 2.0
        elif behavior == "straight" and offset_error < 0.1:
            progress_reward *= 1.2

        steering_penalty = -0.05 * abs(action)
        smoothness_penalty = -0.2 * abs(action - self.last_action)
        boundary_penalty = -2.0 if abs(self.offset) > (self.track_width / 2.2) * 0.95 else 0.0

        total_reward = (
                progress_reward +
                offset_reward +
                steering_penalty +
                smoothness_penalty +
                boundary_penalty
        )

        if self.logging_enabled:
            with open(self.log_file, "a") as f:
                f.write(
                    f"pos={self.position}, stage={behavior}, norm_offset={norm_offset:.2f}, desired={desired_offset:.2f}, "
                    f"offset_error={offset_error:.2f}, offset_reward={offset_reward:.2f}, "
                    f"progress_reward={progress_reward:.2f}, steering_penalty={steering_penalty:.2f}, "
                    f"smoothness_penalty={smoothness_penalty:.2f}, boundary_penalty={boundary_penalty:.2f}, "
                    f"total={total_reward:.2f}\n"
                )

        return total_reward


    def _precompute_curvatures(self):
        raw_curvs = [self._compute_curvature(i) for i in range(self.track_length)]
        window = 5
        smooth_curvs = np.convolve(raw_curvs, np.ones(window) / window, mode='same')
        return smooth_curvs.tolist()

    def _detect_curve_segments(self, threshold=0.003):
        segments = []
        point_behaviors = ["straight"] * len(self.curvatures)
        in_curve = False
        start_idx = 0
        direction = 0
        extend_after = 10
        extend_counter = 0

        for i, curv in enumerate(self.curvatures):
            if abs(curv) > threshold:
                if not in_curve:
                    in_curve = True
                    start_idx = i  # 🔥 no shifting! take exactly where curve starts
                    direction = -1 if curv > 0 else 1
                extend_counter = extend_after  # reset extend counter whenever curve is strong
            else:
                if in_curve:
                    if extend_counter > 0:
                        extend_counter -= 1
                    else:
                        end_idx = i - 1
                        self._label_curve(point_behaviors, start_idx, end_idx, direction)
                        segments.append((start_idx, end_idx, direction))
                        in_curve = False

        if in_curve:
            end_idx = len(self.curvatures) - 1
            self._label_curve(point_behaviors, start_idx, end_idx, direction)
            segments.append((start_idx, end_idx, direction))

        self.point_behaviors = point_behaviors
        # 👇 Force first few points to be adjust/straight (already exists)
        FORCE_ADJUST_POINTS = 20
        for i in range(FORCE_ADJUST_POINTS):
            if i < len(self.point_behaviors):
                if abs(self.curvatures[i]) > 0.003:
                    direction = -1 if self.curvatures[i] > 0 else 1
                    self.point_behaviors[i] = ("adjust", direction)
                else:
                    self.point_behaviors[i] = "straight"

        for i in range(1, FORCE_ADJUST_POINTS + 1):
            idx = len(self.point_behaviors) - i
            if idx >= 0:
                if abs(self.curvatures[idx]) > 0.003:
                    direction = -1 if self.curvatures[idx] > 0 else 1
                    self.point_behaviors[idx] = ("adjust", direction)
                else:
                    self.point_behaviors[idx] = "straight"

        return segments

    def _label_curve(self, behaviors, start_idx, end_idx, direction):
        curve_len = end_idx - start_idx + 1

        if curve_len <= 5:
            for j in range(start_idx, end_idx + 1):
                behaviors[j] = ("apex", direction)
            return

        # 1. Find apex (maximum curvature)
        apex_idx = start_idx + np.argmax(np.abs(self.curvatures[start_idx:end_idx + 1]))

        # 2. Define apex zone (~20% of curve length)
        apex_zone_size = max(3, int(curve_len * 0.3))  # 🔥 Apex = 20% of curve
        apex_half = apex_zone_size // 2

        apex_start = max(start_idx, apex_idx - apex_half)
        apex_end = min(end_idx, apex_idx + apex_half)

        # 3. Remaining points for entry and exit
        before_apex_len = apex_start - start_idx
        after_apex_len = end_idx - apex_end

        # 4. Adjust lengths (equal to entry/exit)
        adjust_len_entry = int(before_apex_len / 2)
        adjust_len_exit = int(after_apex_len / 2)

        # 5. Compute adjust zones
        adjust_start_entry = max(0, start_idx - adjust_len_entry)
        adjust_end_entry = start_idx - 1

        adjust_start_exit = end_idx + 1
        adjust_end_exit = min(self.track_length - 1, end_idx + adjust_len_exit)

        # 6. Check overlap with previous curve
        if adjust_end_entry >= adjust_start_entry:
            for j in range(adjust_start_entry, adjust_end_entry + 1):
                if behaviors[j] != "straight":
                    mid = (adjust_start_entry + adjust_end_entry) // 2
                    if j <= mid:
                        behaviors[j] = ("adjust", direction)
                    else:
                        behaviors[j] = ("entry", direction)
                else:
                    behaviors[j] = ("adjust", direction)

        # 7. Check overlap with next curve
        if adjust_end_exit >= adjust_start_exit:
            for j in range(adjust_start_exit, adjust_end_exit + 1):
                if behaviors[j] != "straight":
                    mid = (adjust_start_exit + adjust_end_exit) // 2
                    if j <= mid:
                        behaviors[j] = ("exit", direction)
                    else:
                        behaviors[j] = ("adjust", direction)
                else:
                    behaviors[j] = ("adjust", direction)

        # 8. Label the actual curve
        for j in range(start_idx, end_idx + 1):
            if start_idx <= j < apex_start:
                behaviors[j] = ("entry", direction)
            elif apex_start <= j <= apex_end:
                behaviors[j] = ("apex", direction)
            elif apex_end < j <= end_idx:
                behaviors[j] = ("exit", direction)

    def _find_closest_index(self, distance):
        idx = np.searchsorted(self.arc_lengths, distance)
        return np.clip(idx, 0, self.track_length - 1)

    def _index_at_distance(self, start_idx, distance):
        target_distance = self.arc_lengths[start_idx] + distance
        for i in range(start_idx, len(self.arc_lengths)):
            if self.arc_lengths[i] >= target_distance:
                return i
        return len(self.arc_lengths) - 1

    def _compute_arc_lengths(self, x, y):
        distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        return np.concatenate(([0], np.cumsum(distances)))

    def _compute_curvature(self, idx):
        if idx <= 0 or idx >= self.track_length - 1:
            return 0.0
        dx = (self.center_x[idx + 1] - self.center_x[idx - 1]) / 2
        dy = (self.center_y[idx + 1] - self.center_y[idx - 1]) / 2
        ddx = self.center_x[idx + 1] - 2 * self.center_x[idx] + self.center_x[idx - 1]
        ddy = self.center_y[idx + 1] - 2 * self.center_y[idx] + self.center_y[idx - 1]
        num = abs(dx * ddy - dy * ddx)
        denom = (dx ** 2 + dy ** 2 + 1e-8) ** 1.5
        return num / denom
