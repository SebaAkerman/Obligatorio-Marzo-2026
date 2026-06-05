"""
Experimento: Impacto de Alpha-Beta Pruning en Isolation.

Compara tres variantes de Minimax:
  1. Minimax puro (sin AB, sin move ordering)
  2. Minimax + Alpha-Beta Pruning (sin move ordering)
  3. Minimax + Alpha-Beta Pruning + Move Ordering

Métricas: nodos expandidos promedio por partida, tiempo por partida, win rate.

Uso (desde Isolation/):
    poetry run python ../scripts/experiment_ab_impact.py
"""

import sys, time, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Isolation"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from isolation_env import IsolationEnv
from minimax_agent import MinimaxAgent
from random_agent import RandomAgent
from heuristics import eval_mobility_only

os.makedirs(str(Path(__file__).parent.parent / "reports/figures"), exist_ok=True)
os.makedirs(str(Path(__file__).parent.parent / "models/mate"), exist_ok=True)

FIGURES = Path(__file__).parent.parent / "reports/figures"
MODELS = Path(__file__).parent.parent / "models/mate"

N_GAMES = 50
DEPTHS = [2, 3, 4]


def run_experiment(agent_factory, n_games, label):
    wins, total_nodes, times = 0, [], []
    for _ in range(n_games):
        env = IsolationEnv()
        board = env.reset()
        agent = agent_factory()
        random_opp = RandomAgent(2)
        done = False
        t0 = time.time()
        nodes_this_game = 0
        while not done:
            if env.current_player == 1:
                action = agent.next_action(board)
                nodes_this_game += agent._nodes_expanded
            else:
                action = random_opp.next_action(board)
            if action is None:
                winner = env.current_player % 2 + 1
                wins += int(winner == 1)
                break
            board, _, done, winner, _ = env.step(action)
            if done:
                wins += int(winner == 1)
        times.append(time.time() - t0)
        total_nodes.append(nodes_this_game)
    return {
        "label": label,
        "win_rate": wins / n_games,
        "avg_nodes": np.mean(total_nodes),
        "avg_time_s": np.mean(times),
        "std_nodes": np.std(total_nodes),
    }


results = []

for depth in DEPTHS:
    print(f"\n=== Depth {depth} ({N_GAMES} games each) ===")

    configs = [
        ("Pure Minimax",
         lambda d=depth: MinimaxAgent(1, d, eval_mobility_only, use_alpha_beta=False, use_move_ordering=False)),
        ("AB Pruning",
         lambda d=depth: MinimaxAgent(1, d, eval_mobility_only, use_alpha_beta=True, use_move_ordering=False)),
        ("AB + Move Ordering",
         lambda d=depth: MinimaxAgent(1, d, eval_mobility_only, use_alpha_beta=True, use_move_ordering=True)),
    ]

    for label, factory in configs:
        print(f"  {label}...", end=" ", flush=True)
        r = run_experiment(factory, N_GAMES, label)
        r["depth"] = depth
        results.append(r)
        print(f"nodes={r['avg_nodes']:.0f} time={r['avg_time_s']:.3f}s win={r['win_rate']:.0%}")

df = pd.DataFrame(results)
df.to_csv(str(MODELS / "ab_impact_results.csv"), index=False)
print("\n\n=== TABLA COMPLETA ===")
print(df[["depth","label","avg_nodes","avg_time_s","win_rate"]].to_string(index=False))

# ---------- Figura ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {"Pure Minimax": "#e74c3c", "AB Pruning": "#3498db", "AB + Move Ordering": "#2ecc71"}
markers = {"Pure Minimax": "o", "AB Pruning": "s", "AB + Move Ordering": "^"}

for label in ["Pure Minimax", "AB Pruning", "AB + Move Ordering"]:
    sub = df[df["label"] == label]
    axes[0].plot(sub["depth"], sub["avg_nodes"],
                 color=colors[label], marker=markers[label], lw=2, label=label)
    axes[1].plot(sub["depth"], sub["avg_time_s"],
                 color=colors[label], marker=markers[label], lw=2, label=label)

axes[0].set_yscale("log")
axes[0].set_xlabel("Profundidad de búsqueda")
axes[0].set_ylabel("Nodos expandidos (promedio, escala log)")
axes[0].set_title("Impacto de Alpha-Beta: Nodos Expandidos")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xticks(DEPTHS)

axes[1].set_yscale("log")
axes[1].set_xlabel("Profundidad de búsqueda")
axes[1].set_ylabel("Tiempo promedio por partida (s, escala log)")
axes[1].set_title("Impacto de Alpha-Beta: Tiempo de Ejecución")
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xticks(DEPTHS)

fig.suptitle(f"Alpha-Beta Pruning vs Minimax Puro — {N_GAMES} partidas por configuración",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(str(FIGURES / "ab_impact.png"), dpi=150, bbox_inches="tight")
print(f"\nFigura guardada: reports/figures/ab_impact.png")

# ---------- Tabla de reducción ----------
print("\n=== TABLA: REDUCCIÓN DE NODOS vs PURE MINIMAX ===")
for depth in DEPTHS:
    base = df[(df["depth"] == depth) & (df["label"] == "Pure Minimax")]["avg_nodes"].values[0]
    for label in ["AB Pruning", "AB + Move Ordering"]:
        val = df[(df["depth"] == depth) & (df["label"] == label)]["avg_nodes"].values[0]
        red = (base - val) / base * 100
        print(f"  depth={depth} | {label}: {val:.0f} nodos ({red:.1f}% reducción vs Pure)")
