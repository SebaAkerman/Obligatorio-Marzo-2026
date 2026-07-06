"""Discretización de estados/acciones para MountainCarContinuous-v0.

Observation space: Box(2,) -> [posición x, velocidad], con x en [-1.2, 0.6]
y vel en [-0.07, 0.07]. Action space: Box(1,) -> fuerza en [-1.0, 1.0].
Para poder usar Q-Learning tabular hay que discretizar ambos.
"""

import numpy as np


class Discretizer:

    POS_MIN, POS_MAX = -1.2, 0.6
    VEL_MIN, VEL_MAX = -0.07, 0.07
    ACT_MIN, ACT_MAX = -1.0, 1.0

    def __init__(
        self,
        n_pos_bins: int = 20,
        n_vel_bins: int = 20,
        n_actions: int = 10,
    ) -> None:
        self.n_pos_bins = n_pos_bins
        self.n_vel_bins = n_vel_bins
        self.n_actions = n_actions

        self.pos_bins = np.linspace(self.POS_MIN, self.POS_MAX, n_pos_bins)
        self.vel_bins = np.linspace(self.VEL_MIN, self.VEL_MAX, n_vel_bins)
        self.actions = np.linspace(self.ACT_MIN, self.ACT_MAX, n_actions)

    def obs_to_state(self, obs: np.ndarray) -> tuple[int, int]:
        x, vel = obs
        x_bin = int(np.digitize(x, self.pos_bins))
        vel_bin = int(np.digitize(vel, self.vel_bins))
        return x_bin, vel_bin

    @property
    def state_shape(self) -> tuple[int, int]:
        return (self.n_pos_bins + 1, self.n_vel_bins + 1)

    def action_index_to_continuous(self, idx: int) -> np.ndarray:
        return np.array([self.actions[idx]])

    def sample_action_index(self) -> int:
        return np.random.randint(self.n_actions)

    def __repr__(self) -> str:
        return (
            f"Discretizer(n_pos_bins={self.n_pos_bins}, "
            f"n_vel_bins={self.n_vel_bins}, "
            f"n_actions={self.n_actions})"
        )
