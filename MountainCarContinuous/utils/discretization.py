"""
Módulo de discretización para MountainCarContinuous-v0.

MountainCarContinuous tiene:
  - Observation space: Box(2,) → [posición x, velocidad]
      x      ∈ [-1.2,  0.6]
      vel    ∈ [-0.07, 0.07]
  - Action space: Box(1,) → fuerza ∈ [-1.0, 1.0]  (continua)

Para aplicar Q-Learning tabular se deben discretizar ambos espacios.
"""

import numpy as np


class Discretizer:
    """
    Discretiza observaciones y acciones del ambiente MountainCarContinuous.

    Parámetros
    ----------
    n_pos_bins : int
        Número de bins para la posición x.
    n_vel_bins : int
        Número de bins para la velocidad.
    n_actions : int
        Número de acciones discretas uniformes en [-1, 1].
    """

    # Límites del observation space de MountainCarContinuous-v0
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

        # Grillas de discretización
        self.pos_bins = np.linspace(self.POS_MIN, self.POS_MAX, n_pos_bins)
        self.vel_bins = np.linspace(self.VEL_MIN, self.VEL_MAX, n_vel_bins)
        self.actions = np.linspace(self.ACT_MIN, self.ACT_MAX, n_actions)

    # ------------------------------------------------------------------
    # Observaciones
    # ------------------------------------------------------------------

    def obs_to_state(self, obs: np.ndarray) -> tuple[int, int]:
        """Convierte una observación continua en un índice de estado discreto."""
        x, vel = obs
        x_bin = int(np.digitize(x, self.pos_bins))
        vel_bin = int(np.digitize(vel, self.vel_bins))
        return x_bin, vel_bin

    @property
    def state_shape(self) -> tuple[int, int]:
        """Forma de la dimensión de estados en la tabla Q."""
        return (self.n_pos_bins + 1, self.n_vel_bins + 1)

    # ------------------------------------------------------------------
    # Acciones
    # ------------------------------------------------------------------

    def action_index_to_continuous(self, idx: int) -> np.ndarray:
        """Convierte un índice de acción discreta en la acción continua para el ambiente."""
        return np.array([self.actions[idx]])

    def sample_action_index(self) -> int:
        """Devuelve un índice de acción aleatorio."""
        return np.random.randint(self.n_actions)

    def __repr__(self) -> str:
        return (
            f"Discretizer(n_pos_bins={self.n_pos_bins}, "
            f"n_vel_bins={self.n_vel_bins}, "
            f"n_actions={self.n_actions})"
        )
