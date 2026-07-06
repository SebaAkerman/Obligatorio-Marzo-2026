import pickle
import random
from pathlib import Path

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from utils.discretization import Discretizer


class QLearningAgent:
    # Q-Learning tabular con política epsilon-greedy

    def __init__(
        self,
        n_pos_bins: int = 20,
        n_vel_bins: int = 20,
        n_actions: int = 10,
        q_init: float = 0.0,
    ) -> None:
        self.disc = Discretizer(
            n_pos_bins=n_pos_bins,
            n_vel_bins=n_vel_bins,
            n_actions=n_actions,
        )

        self.q_init = q_init
        # q_init > 0 es inicialización optimista, ayuda a explorar al principio
        self.Q = np.full((*self.disc.state_shape, self.disc.n_actions), q_init)

        # se pisan en train_agent
        self.alpha: float = 0.1
        self.gamma: float = 0.99
        self.epsilon: float = 1.0

        self.training_rewards: list[float] = []

    def next_action(self, obs: np.ndarray) -> np.ndarray:
        state = self.disc.obs_to_state(obs)

        if random.random() < self.epsilon:
            action_idx = self.disc.sample_action_index()   # exploración
        else:
            action_idx = int(np.argmax(self.Q[state]))     # explotación

        return self.disc.action_index_to_continuous(action_idx)

    def _action_index(self, obs: np.ndarray) -> int:
        """Igual que next_action pero devuelve el índice, para poder actualizar Q."""
        state = self.disc.obs_to_state(obs)
        if random.random() < self.epsilon:
            return self.disc.sample_action_index()
        return int(np.argmax(self.Q[state]))

    def _update(
        self,
        obs: np.ndarray,
        action_idx: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        state = self.disc.obs_to_state(obs)
        next_state = self.disc.obs_to_state(next_obs)

        max_next_q = 0.0 if done else float(np.max(self.Q[next_state]))
        target = reward + self.gamma * max_next_q
        self.Q[state][action_idx] += self.alpha * (target - self.Q[state][action_idx])

    def train_agent(
        self,
        env: gym.Env,
        episodes: int = 1000,
        epsilon: float = 1.0,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        max_steps: int = 999,
        verbose: bool = True,
        log_every: int = 200,
    ) -> list[float]:
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

                self._update(obs, action_idx, reward, next_obs, done)
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
                    f"ε: {self.epsilon:.4f}"
                )

        return self.training_rewards

    def test_agent(
        self,
        env: gym.Env,
        episodes: int = 10,
        max_steps: int = 999,
        render: bool = False,
    ) -> dict[str, float]:
        """Evalúa con política greedy pura (epsilon = 0)."""
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

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "Q": self.Q,
                    "disc": self.disc,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "training_rewards": self.training_rewards,
                    "q_init": self.q_init,
                },
                f,
            )
        print(f"Modelo guardado en: {path}")

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        disc: Discretizer = data["disc"]
        agent = cls(
            n_pos_bins=disc.n_pos_bins,
            n_vel_bins=disc.n_vel_bins,
            n_actions=disc.n_actions,
        )
        agent.Q = data["Q"]
        agent.alpha = data["alpha"]
        agent.gamma = data["gamma"]
        agent.epsilon = data["epsilon"]
        agent.training_rewards = data.get("training_rewards", [])
        agent.q_init = data.get("q_init", 0.0)
        print(f"Modelo cargado desde: {path}")
        return agent
