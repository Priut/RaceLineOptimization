import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, offset_bins=6, curvature_bins=7, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.offset_bins = offset_bins
        self.curvature_bins = curvature_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Steering actions (steering deltas)
        self.actions = [-1, 0, 1]  # steer left, stay, steer right

        # Q-table indexed by (curv_bin, offset_bin)
        self.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def _discretize_state(self, state):
        """
        Discretizes the extended state:
        (normalized_pos, norm_offset, offset_velocity, steering_angle, curvature_now, curvature_5, curvature_10)
        """
        pos_bin = int(state[0] * 10)  # 10 bins for position [0, 1]

        offset_bin = int((state[1] + 1) * 5)  # [-1, 1] → [0, 10]
        offset_bin = np.clip(offset_bin, 0, 10)

        # Prioritized discretization for offset velocity
        vel = np.clip(state[2], -3.0, 3.0)  # most values fall here
        vel_bin = int((vel + 3) / 6 * 5)  # [-3, 3] → [0, 5]
        vel_bin = np.clip(vel_bin, 0, 5)

        # Prioritized discretization for steering angle
        steer = np.clip(state[3], -0.5, 0.5)  # steering is limited by environment
        steer_bin = int((steer + 0.5) / 1.0 * 5)  # [-0.5, 0.5] → [0, 5]
        steer_bin = np.clip(steer_bin, 0, 5)

        # Curvature now, +5, +10 (nonlinear binning works fine for these)
        curv_now_bin = min(int(state[4] * self.curvature_bins), self.curvature_bins - 1)
        curv_5_bin = min(int(state[5] * self.curvature_bins), self.curvature_bins - 1)
        curv_10_bin = min(int(state[6] * self.curvature_bins), self.curvature_bins - 1)

        return pos_bin, offset_bin, vel_bin, steer_bin, curv_now_bin, curv_5_bin, curv_10_bin

    def choose_action(self, state):
        discrete_state = self._discretize_state(state)
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        q_values = self.q_table[discrete_state]
        return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state, done):
        s = self._discretize_state(state)
        s_next = self._discretize_state(next_state)

        q_current = self.q_table[s][action]
        q_max_next = max(self.q_table[s_next].values()) if not done else 0.0

        self.q_table[s][action] += self.alpha * (reward + self.gamma * q_max_next - q_current)

        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
