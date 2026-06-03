"""
Script de entrenamiento para el Proyecto LOST.

Uso (ejecutar desde la raíz del repo):
    cd MountainCarContinuous
    poetry run python ../scripts/train_lost.py --agent qlearning --episodes 5000
    poetry run python ../scripts/train_lost.py --agent dynaq --n-planning 10
    poetry run python ../scripts/train_lost.py --agent qlearning --pos-bins 30 --vel-bins 30 --n-actions 15

Ver todas las opciones:
    poetry run python ../scripts/train_lost.py --help
"""

import argparse
import sys
import time
from pathlib import Path

# El script se ejecuta desde MountainCarContinuous/, que es donde están los módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "MountainCarContinuous"))

import gymnasium as gym

from utils.discretization import Discretizer
from q_learning_agent import QLearningAgent
from dyna_q_agent import DynaQAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de agentes para MountainCarContinuous-v0"
    )
    parser.add_argument("--agent", choices=["qlearning", "dynaq"], default="qlearning")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--pos-bins", type=int, default=20)
    parser.add_argument("--vel-bins", type=int, default=20)
    parser.add_argument("--n-actions", type=int, default=10)
    parser.add_argument("--n-planning", type=int, default=10, help="Solo para Dyna-Q")
    parser.add_argument("--q-init", type=float, default=0.0, help="Inicialización optimista de Q (>0 = optimista)")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make("MountainCarContinuous-v0", render_mode=None)

    if args.agent == "qlearning":
        agent = QLearningAgent(
            n_pos_bins=args.pos_bins,
            n_vel_bins=args.vel_bins,
            n_actions=args.n_actions,
            q_init=args.q_init,
        )
        print(f"Agente: Q-Learning | {agent.disc}")
    else:
        agent = DynaQAgent(
            n_pos_bins=args.pos_bins,
            n_vel_bins=args.vel_bins,
            n_actions=args.n_actions,
            n_planning_steps=args.n_planning,
            q_init=args.q_init,
        )
        print(f"Agente: Dyna-Q (n_planning={args.n_planning}) | {agent.disc}")

    print(f"Iniciando entrenamiento: {args.episodes} episodios...")
    t0 = time.time()

    rewards = agent.train_agent(
        env,
        episodes=args.episodes,
        epsilon=args.epsilon,
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )

    elapsed = time.time() - t0
    print(f"Entrenamiento completado en {elapsed:.1f}s")

    print("\nEvaluando agente...")
    results = agent.test_agent(env, episodes=args.eval_episodes)
    env.close()

    save_path = args.save_path
    if not save_path:
        model_dir = Path(__file__).parent.parent / "models" / "lost"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(
            model_dir
            / f"{args.agent}_pos{args.pos_bins}_vel{args.vel_bins}_act{args.n_actions}.pkl"
        )
    agent.save(save_path)


if __name__ == "__main__":
    main()