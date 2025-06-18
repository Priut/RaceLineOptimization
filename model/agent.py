import numpy as np
import random
from collections import defaultdict
import pickle

class QLearningAgent:
    """
    A Q-learning agent for learning steering control on a racing track.

    Attributes:
        offset_bins (int): Number of bins for discretizing lateral offset.
        curvature_bins (int): Number of bins for discretizing curvature.
        alpha (float): Learning rate.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Rate at which epsilon decays after each episode.
        epsilon_min (float): Minimum value for epsilon.
        actions (List[float]): Set of possible steering angle changes.
        q_table (defaultdict): Q-table mapping discretized states to action-values.
    """

    def __init__(self, offset_bins=8, curvature_bins=8, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.98, epsilon_min=0.05):
        self.offset_bins = offset_bins
        self.curvature_bins = curvature_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Steering control actions (from hard left to hard right)
        self.actions = [-0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5]

        # Q-table with default value 0.0 for all actions in unseen states
        self.q_table = defaultdict(self._q_value_initializer)

    def _q_value_initializer(self):
        return {a: 0.0 for a in self.actions}

    def _discretize_state(self, state):
        """
        Converts the continuous environment state into a discrete tuple used for indexing the Q-table.
        """
        normalized_pos, norm_offset, offset_velocity, steering_angle, curvature_now, curvature_5, curvature_10 = state

        # Discretize progress along the track
        pos_bin = np.clip(int(normalized_pos * 10), 0, 9)

        # Discretize lateral offset
        offset_bin = np.clip(int((norm_offset + 1.0) * (self.offset_bins / 2)), 0, self.offset_bins - 1)

        # Discretize steering angle
        steering_bin = np.clip(int((steering_angle + 0.5) * 5), 0, 5)

        # Discretize curvature values
        curv_scale = 20.0
        curvature_now_bin = int(np.clip(curvature_now * curv_scale, 0, self.curvature_bins - 1))
        curvature_5_bin = int(np.clip(curvature_5 * curv_scale, 0, self.curvature_bins - 1))
        curvature_10_bin = int(np.clip(curvature_10 * curv_scale, 0, self.curvature_bins - 1))

        return (pos_bin, offset_bin, steering_bin, curvature_now_bin, curvature_5_bin, curvature_10_bin)

    def choose_action(self, state):
        """
        Chooses an action using epsilon-greedy policy. In curves, exploration is boosted.
        """
        discrete_state = self._discretize_state(state)
        curvature_now = state[4]

        # Increase exploration in curves
        epsilon_local = min(self.epsilon * 2 if abs(curvature_now) > 0.01 else self.epsilon, 1.0)

        if random.random() < epsilon_local:
            return random.choice(self.actions)

        # Greedy action selection
        q_values = self.q_table[discrete_state]
        return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state, done):
        """
        Performs the Q-learning update based on the observed transition.
        """
        s = self._discretize_state(state)
        s_next = self._discretize_state(next_state)

        q_current = self.q_table[s][action]
        q_max_next = max(self.q_table[s_next].values()) if not done else 0.0

        # Standard Q-learning formula
        self.q_table[s][action] += self.alpha * (reward + self.gamma * q_max_next - q_current)

        if done:
            # Decay epsilon and learning rate at end of episode
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.alpha = max(self.alpha * 0.99, 0.01)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
