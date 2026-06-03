"""
Script de evaluación y torneo para el Proyecto MATE (Isolation).

Uso (ejecutar desde la carpeta Isolation/):
    cd Isolation
    poetry run python ../scripts/evaluate_mate.py --n-games 100 --depth 3
    poetry run python ../scripts/evaluate_mate.py --agent1 minimax --agent2 random --n-games 50
    poetry run python ../scripts/evaluate_mate.py --agent1 minimax --agent2 expectimax --no-alpha-beta

Ver todas las opciones:
    poetry run python ../scripts/evaluate_mate.py --help
"""

import argparse
import sys
import time
from pathlib import Path

# El script se ejecuta desde Isolation/, que es donde están los módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "Isolation"))

from isolation_env import IsolationEnv
from random_agent import RandomAgent
from minimax_agent import MinimaxAgent
from expectimax_agent import ExpectimaxAgent
from heuristics import HEURISTICS


def run_tournament(agent1, agent2, n_games: int = 100) -> dict:
    """
    Ejecuta n_games partidas entre agent1 (jugador 1) y agent2 (jugador 2).
    Devuelve estadísticas de resultados.
    """
    wins_agent1 = 0
    wins_agent2 = 0
    total_moves = []
    nodes_agent1 = []
    nodes_agent2 = []

    for _ in range(n_games):
        env = IsolationEnv()
        board = env.reset()
        done = False
        moves = 0

        while not done:
            current_player = env.current_player

            if current_player == 1:
                action = agent1.next_action(board)
            else:
                action = agent2.next_action(board)

            if action is None:
                winner = current_player % 2 + 1
                wins_agent1 += int(winner == 1)
                wins_agent2 += int(winner == 2)
                break

            board, _, done, winner, _ = env.step(action)
            moves += 1

            if done:
                wins_agent1 += int(winner == 1)
                wins_agent2 += int(winner == 2)

        total_moves.append(moves)

        if hasattr(agent1, "_nodes_expanded"):
            nodes_agent1.append(agent1._nodes_expanded)
        if hasattr(agent2, "_nodes_expanded"):
            nodes_agent2.append(agent2._nodes_expanded)

    return {
        "wins_agent1": wins_agent1,
        "wins_agent2": wins_agent2,
        "win_rate_agent1": wins_agent1 / n_games,
        "win_rate_agent2": wins_agent2 / n_games,
        "avg_moves": sum(total_moves) / len(total_moves),
        "avg_nodes_agent1": sum(nodes_agent1) / len(nodes_agent1) if nodes_agent1 else None,
        "avg_nodes_agent2": sum(nodes_agent2) / len(nodes_agent2) if nodes_agent2 else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación de agentes para Isolation")
    parser.add_argument(
        "--agent1", choices=["minimax", "expectimax", "random"], default="minimax"
    )
    parser.add_argument(
        "--agent2", choices=["minimax", "expectimax", "random"], default="random"
    )
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--heuristic", choices=list(HEURISTICS.keys()), default="mobility_only"
    )
    parser.add_argument("--alpha-beta", action="store_true", default=True)
    parser.add_argument("--no-alpha-beta", dest="alpha_beta", action="store_false")
    parser.add_argument("--output", type=str, default=None, help="Guardar resultados en CSV")
    return parser.parse_args()


def build_agent(
    agent_type: str, player: int, depth: int, heuristic_name: str, alpha_beta: bool
):
    heuristic = HEURISTICS[heuristic_name]
    if agent_type == "minimax":
        return MinimaxAgent(
            player=player,
            max_depth=depth,
            heuristic=heuristic,
            use_alpha_beta=alpha_beta,
        )
    elif agent_type == "expectimax":
        return ExpectimaxAgent(player=player, max_depth=depth, heuristic=heuristic)
    elif agent_type == "random":
        return RandomAgent(player=player)
    raise ValueError(f"Agente desconocido: {agent_type}")


def main() -> None:
    args = parse_args()

    print("Configuración del torneo:")
    print(f"  Agent1 : {args.agent1} (Jugador 1)")
    print(f"  Agent2 : {args.agent2} (Jugador 2)")
    print(f"  Profundidad : {args.depth}")
    print(f"  Heurística  : {args.heuristic}")
    print(f"  Alpha-Beta  : {args.alpha_beta}")
    print(f"  Partidas    : {args.n_games}\n")

    agent1 = build_agent(args.agent1, 1, args.depth, args.heuristic, args.alpha_beta)
    agent2 = build_agent(args.agent2, 2, args.depth, args.heuristic, args.alpha_beta)

    t0 = time.time()
    results = run_tournament(agent1, agent2, n_games=args.n_games)
    elapsed = time.time() - t0

    print("=== RESULTADOS ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\nTiempo total: {elapsed:.2f}s")

    if args.output:
        import pandas as pd

        pd.DataFrame(
            [
                {
                    **results,
                    "agent1": args.agent1,
                    "agent2": args.agent2,
                    "depth": args.depth,
                    "heuristic": args.heuristic,
                    "alpha_beta": args.alpha_beta,
                    "n_games": args.n_games,
                }
            ]
        ).to_csv(args.output, index=False)
        print(f"Resultados guardados en: {args.output}")


if __name__ == "__main__":
    main()