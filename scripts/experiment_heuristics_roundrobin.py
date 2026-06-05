"""
Experimento: Round-Robin completo de heurísticas en Isolation.

Enfrenta TODAS las heurísticas entre sí (round-robin) para determinar
cuál es la más fuerte. Incluye h_territory (BFS) y h_future_mobility.

Métricas: puntos totales en round-robin, IC 95%, win rate vs cada oponente.

Uso (desde Isolation/):
    poetry run python ../scripts/experiment_heuristics_roundrobin.py
"""

import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Isolation"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from isolation_env import IsolationEnv
from minimax_agent import MinimaxAgent
from heuristics import (
    eval_mobility_only, eval_mobility_center, eval_full,
    eval_territory, eval_mobility_territory,
)

FIGURES = Path(__file__).parent.parent / "reports/figures"
MODELS = Path(__file__).parent.parent / "models/mate"
os.makedirs(str(FIGURES), exist_ok=True)
os.makedirs(str(MODELS), exist_ok=True)

DEPTH = 3
N_GAMES = 30  # por sentido (60 total entre cada par: 30 como P1 + 30 como P2)

HEURISTICS = {
    "mob_only":       eval_mobility_only,
    "mob_center":     eval_mobility_center,
    "full":           eval_full,
    "territory":      eval_territory,
    "mob_territory":  eval_mobility_territory,
}
NAMES = list(HEURISTICS.keys())


def play_match(h1_name, h2_name, n_games):
    """h1 como P1, h2 como P2. Devuelve (wins_h1, wins_h2)."""
    wins = {1: 0, 2: 0}
    for _ in range(n_games):
        env = IsolationEnv()
        board = env.reset()
        a1 = MinimaxAgent(1, DEPTH, HEURISTICS[h1_name], use_alpha_beta=True)
        a2 = MinimaxAgent(2, DEPTH, HEURISTICS[h2_name], use_alpha_beta=True)
        done = False
        while not done:
            cur = env.current_player
            action = a1.next_action(board) if cur == 1 else a2.next_action(board)
            if action is None:
                wins[cur % 2 + 1] += 1
                break
            board, _, done, winner, _ = env.step(action)
            if done:
                wins[winner] += 1
    return wins[1], wins[2]


# Matriz de victorias: win_matrix[i][j] = victorias de i contra j (como P1 + como P2)
n = len(NAMES)
win_matrix = np.zeros((n, n), dtype=int)
total_games = np.zeros((n, n), dtype=int)

print(f"Round-Robin: {n} heurísticas × {N_GAMES*2} partidas cada par\n")

for i, h1 in enumerate(NAMES):
    for j, h2 in enumerate(NAMES):
        if i == j:
            continue
        print(f"  {h1} vs {h2}...", end=" ", flush=True)
        # h1 como P1
        w1, w2 = play_match(h1, h2, N_GAMES)
        win_matrix[i][j] += w1
        win_matrix[j][i] += w2
        total_games[i][j] += N_GAMES
        total_games[j][i] += N_GAMES
        # h1 como P2 (simetría — evita doble conteo ya cubierto por [j,i])
        print(f"[{h1}:{w1} {h2}:{w2}]")

# Puntos totales (ignorando duplicados de simetría)
# Calculamos winrate normalizado
points = {}
for i, name in enumerate(NAMES):
    total_w = sum(win_matrix[i][j] for j in range(n) if j != i)
    total_g = sum(total_games[i][j] for j in range(n) if j != i)
    win_rate = total_w / total_g if total_g > 0 else 0
    ci95 = 1.96 * np.sqrt(win_rate * (1 - win_rate) / total_g) if total_g > 0 else 0
    points[name] = {"total_wins": total_w, "total_games": total_g,
                    "win_rate": win_rate, "ci95": ci95}

df_points = pd.DataFrame(points).T.sort_values("win_rate", ascending=False)
df_points.to_csv(str(MODELS / "roundrobin_results.csv"))

print("\n=== ROUND-ROBIN FINAL STANDINGS ===")
print(df_points[["total_wins","total_games","win_rate","ci95"]].to_string())

# ---------- Heatmap ----------
# Win rate de fila vs columna
rate_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j and total_games[i][j] > 0:
            rate_matrix[i][j] = win_matrix[i][j] / total_games[i][j]
        elif i == j:
            rate_matrix[i][j] = 0.5  # diagonal: 50% (no jugado)

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.eye(n, dtype=bool)
sns.heatmap(
    rate_matrix, annot=True, fmt=".0%",
    xticklabels=NAMES, yticklabels=NAMES,
    cmap="RdYlGn", vmin=0, vmax=1,
    linewidths=0.5, ax=ax,
    mask=mask,
    cbar_kws={"label": "Win rate (fila vs columna)"},
)
ax.set_title(f"Round-Robin de Heurísticas — depth={DEPTH}, {N_GAMES} partidas/par/rol",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Oponente (columna)")
ax.set_ylabel("Jugador (fila)")
plt.tight_layout()
fig.savefig(str(FIGURES / "heuristic_roundrobin_full.png"), dpi=150, bbox_inches="tight")
print(f"\nFigura: reports/figures/heuristic_roundrobin_full.png")

# ---------- Bar chart de standings ----------
fig2, ax2 = plt.subplots(figsize=(10, 5))
colors_bar = ["#2ecc71" if i == 0 else "#3498db" if i < 3 else "#e74c3c"
              for i in range(len(df_points))]
bars = ax2.barh(df_points.index[::-1], df_points["win_rate"][::-1],
                color=colors_bar[::-1], alpha=0.85)
xerr = df_points["ci95"][::-1]
ax2.errorbar(df_points["win_rate"][::-1], range(len(df_points)),
             xerr=xerr, fmt="none", color="black", capsize=4, lw=1.5)
ax2.axvline(0.5, color="gray", ls="--", alpha=0.6, label="50% (par)")
ax2.set_xlabel("Win Rate (IC 95%)")
ax2.set_title(f"Round-Robin: Ranking Final de Heurísticas (depth={DEPTH})", fontweight="bold")
ax2.grid(alpha=0.3, axis="x")
ax2.legend()
for bar, (name, row) in zip(bars[::-1], df_points.iterrows()):
    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f"{row['win_rate']:.0%} ({int(row['total_wins'])}/{int(row['total_games'])})",
             va="center", fontsize=9)
plt.tight_layout()
fig2.savefig(str(FIGURES / "heuristic_roundrobin_bar.png"), dpi=150, bbox_inches="tight")
print(f"Figura: reports/figures/heuristic_roundrobin_bar.png")
