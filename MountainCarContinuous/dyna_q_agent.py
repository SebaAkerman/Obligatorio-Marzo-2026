"""
Agente Dyna-Q para MountainCarContinuous-v0.

Referencia: Sutton & Barto — Reinforcement Learning: An Introduction
            Capítulos 8.1 y 8.2 (Dyna: Integrated Planning, Acting, and Learning)

Dyna-Q extiende Q-Learning integrando planificación, actuación y aprendizaje:

    Por cada paso real (s, a, r, s'):
        1. Actualizar Q con la experiencia real  ← igual que Q-Learning
        2. Actualizar el modelo: Model(s, a) ← (r, s')
        3. Repetir n veces (planificación simulada):
            - Samplear (s̃, ã) visitado anteriormente
            - Obtener (r̃, s̃') del modelo
            - Actualizar Q con esa experiencia simulada

Esto permite aprender más rápido reutilizando experiencias pasadas.
Con n=0 Dyna-Q se reduce a Q-Learning puro.
"""

import pickle
import random
from pathlib import Path

import numpy as np
import gymnasium as gym

from utils.discretization import Discretizer


class DynaQAgent:
    """
    Agente Dyna-Q tabular. Sigue la misma interfaz que QLearningAgent
    (train_agent / test_agent / next_action) para facilitar la comparación.

    Parámetros de __init__
    ----------------------
    n_pos_bins : int
        Número de bins para discretizar la posición x.
    n_vel_bins : int
        Número de bins para discretizar la velocidad.
    n_actions : int
        Número de acciones discretas uniformes en [-1, 1].
    n_planning_steps : int
        Pasos de planificación simulada por cada paso real (n en Dyna-Q).
    """

    def __init__(
        self,
        n_pos_bins: int = 20,
        n_vel_bins: int = 20,
        n_actions: int = 10,
        n_planning_steps: int = 10,
    ) -> None:
        self.disc = Discretizer(
            n_pos_bins=n_pos_bins,
            n_vel_bins=n_vel_bins,
            n_actions=n_actions,
        )
        self.n_planning = n_planning_steps

        # Tabla Q inicializada en cero
        self.Q = np.zeros((*self.disc.state_shape, self.disc.n_actions))

        # Modelo del ambiente: (estado, acción) → (reward, siguiente_estado, done)
        self.model: dict[tuple, tuple] = {}

        # Registro de pares (s, a) visitados para samplear en planificación
        self._visited: list[tuple] = []

        # Hiperparámetros — se setean en train_agent
        self.alpha: float = 0.1
        self.gamma: float = 0.99
        self.epsilon: float = 1.0

        # Historial de entrenamiento
        self.training_rewards: list[float] = []

    # ------------------------------------------------------------------
    # Política ε-greedy  (misma firma que QLearningAgent)
    # ------------------------------------------------------------------

    def next_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Selecciona una acción con política ε-greedy usando self.epsilon actual.

        Devuelve
        --------
        np.ndarray
            Acción continua lista para pasar a env.step().
        """
        state = self.disc.obs_to_state(obs)

        if random.random() < self.epsilon:
            action_idx = self.disc.sample_action_index()
        else:
            action_idx = int(np.argmax(self.Q[state]))

        return self.disc.action_index_to_continuous(action_idx)

    def _action_index(self, obs: np.ndarray) -> int:
        """Igual que next_action pero devuelve el índice (para actualizar Q)."""
        state = self.disc.obs_to_state(obs)
        if random.random() < self.epsilon:
            return self.disc.sample_action_index()
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Actualización Q
    # ------------------------------------------------------------------

    def _q_update(
        self,
        state: tuple,
        action_idx: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        """Regla de actualización Q-Learning (usada tanto en paso real como simulado)."""
        max_next_q = 0.0 if done else float(np.max(self.Q[next_state]))
        target = reward + self.gamma * max_next_q
        self.Q[state][action_idx] += self.alpha * (target - self.Q[state][action_idx])

    # ------------------------------------------------------------------
    # Paso Dyna-Q completo (real + planificación)
    # ------------------------------------------------------------------

    def _dyna_step(
        self,
        obs: np.ndarray,
        action_idx: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        state = self.disc.obs_to_state(obs)
        next_state = self.disc.obs_to_state(next_obs)

        # 1. Actualización Q con experiencia real
        self._q_update(state, action_idx, reward, next_state, done)

        # 2. Actualizar modelo
        self.model[(state, action_idx)] = (reward, next_state, done)
        if (state, action_idx) not in self._visited:
            self._visited.append((state, action_idx))

        # 3. n pasos de planificación simulada
        for _ in range(self.n_planning):
            if not self._visited:
                break
            s, a = random.choice(self._visited)
            r_sim, s_next_sim, done_sim = self.model[(s, a)]
            self._q_update(s, a, r_sim, s_next_sim, done_sim)

    # ------------------------------------------------------------------
    # Entrenamiento  (misma firma que QLearningAgent)
    # ------------------------------------------------------------------

    def train_agent(
        self,
        env: gym.Env,
        episodes: int = 1000,
        epsilon: float = 0.9,
        gamma: float = 0.9,
        alpha: float = 0.99,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        max_steps: int = 999,
        verbose: bool = True,
        log_every: int = 200,
    ) -> list[float]:
        """
        Entrena el agente con Dyna-Q durante `episodes` episodios.

        Parámetros principales (mismos que QLearningAgent.train_agent)
        ---------------------------------------------------------------
        env, episodes, epsilon, gamma, alpha, epsilon_decay, epsilon_min

        Devuelve
        --------
        list[float]
            Recompensa total por episodio.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.training_rewards = []

        for ep in range(1, episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0

            for _ in range(max_steps):
                action_idx = self._action_index(obs)
                action = self.disc.action_index_to_continuous(action_idx)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self._dyna_step(obs, action_idx, reward, next_obs, done)
                obs = next_obs
                total_reward += reward

                if done:
                    break

            self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)
            self.training_rewards.append(total_reward)

            if verbose and ep % log_every == 0:
                mean_100 = float(np.mean(self.training_rewards[-100:]))
                print(
                    f"Ep {ep:>5}/{episodes} | "
                    f"Reward: {total_reward:>8.2f} | "
                    f"Media-100: {mean_100:>8.2f} | "
                    f"ε: {self.epsilon:.4f} | "
                    f"Modelo: {len(self.model)} pares"
                )

        return self.training_rewards

    # ------------------------------------------------------------------
    # Evaluación  (misma firma que QLearningAgent)
    # ------------------------------------------------------------------

    def test_agent(
        self,
        env: gym.Env,
        episodes: int = 10,
        max_steps: int = 999,
        render: bool = False,
    ) -> dict[str, float]:
        """
        Evalúa el agente con política greedy pura (ε = 0).

        Devuelve
        --------
        dict con mean_reward, std_reward y success_rate.
        """
        saved_epsilon = self.epsilon
        self.epsilon = 0.0

        rewards = []
        successes = 0

        for _ in range(episodes):
            obs, _ = env.reset()
            total_reward = 0.0

            for _ in range(max_steps):
                action = self.next_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

                if render:
                    env.render()

                if done:
                    if terminated:
                        successes += 1
                    break

            rewards.append(total_reward)

        self.epsilon = saved_epsilon

        results = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate": successes / episodes,
        }
        print(
            f"[test_agent] Media: {results['mean_reward']:.2f} | "
            f"Std: {results['std_reward']:.2f} | "
            f"Éxitos: {results['success_rate']:.0%}"
        )
        return results

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Guarda el modelo en formato .pkl."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "Q": self.Q,
                    "model": self.model,
                    "disc": self.disc,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "n_planning": self.n_planning,
                    "epsilon": self.epsilon,
                },
                f,
            )
        print(f"Modelo Dyna-Q guardado en: {path}")

    @classmethod
    def load(cls, path: str) -> "DynaQAgent":
        """Carga un modelo previamente guardado."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        disc: Discretizer = data["disc"]
        agent = cls(
            n_pos_bins=disc.n_pos_bins,
            n_vel_bins=disc.n_vel_bins,
            n_actions=disc.n_actions,
            n_planning_steps=data["n_planning"],
        )
        agent.Q = data["Q"]
        agent.model = data["model"]
        agent.alpha = data["alpha"]
        agent.gamma = data["gamma"]
        agent.epsilon = data["epsilon"]
        print(f"Modelo Dyna-Q cargado desde: {path}")
        return agent