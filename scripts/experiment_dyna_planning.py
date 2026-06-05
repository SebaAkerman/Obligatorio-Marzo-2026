"""
Experimento: Impacto de n_planning_steps en Dyna-Q.

Entrena Dyna-Q con distintos valores de n (0=Q-Learning puro, 1, 5, 10, 20, 50)
y compara curvas de aprendizaje y rendimiento final.

Mismos hiperparámetros en todos para comparación justa:
  - 20×20 bins, 15 acciones
  - α=0.2, γ=0.99, ε₀=1.0, decay=0.9995, ε_min=0.05
  - q_init=0.0 (sin inicialización optimista — Dyna-Q no la necesita)
  - 3000 episodios (suficiente para ver diferencias claras entre valores de n)

Uso (desde MountainCarContinuous/):
    poetry run python ../scripts/experiment_dyna_planning.py
"""

import sys, os, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "MountainCarContinuous"))
os.chdir(str(Path(__file__).parent.parent / "MountainCarContinuous"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym

from dyna_q_agent import DynaQAgent

FIGURES = Path(__file__).parent.parent / "reports/figures"
MODELS = Path(__file__).parent.parent / "models/lost"
os.makedirs(str(FIGURES), exist_ok=True)

N_PLANNING_VALUES = [0, 1, 5, 10, 20, 50]
EPISODES = 3000
EVAL_EPISODES = 50

TRAIN_KWARGS = dict(
    alpha=0.2, gamma=0.99, epsilon=1.0,
    epsilon_decay=0.9995, epsilon_min=0.05,
    episodes=EPISODES, verbose=False,
)

colors = {0: "#95a5a6", 1: "#e74c3c", 5: "#e67e22",
          10: "#f1c40f", 20: "#2ecc71", 50: "#3498db"}

results = []
all_rewards = {}

print(f"DynaQ n_planning sweep — {EPISODES} eps, eval {EVAL_EPISODES} eps\n")

for n in N_PLANNING_VALUES:
    label = f"n={n}" + (" (Q-Learning puro)" if n == 0 else "")
    print(f"  Training {label}...", end=" ", flush=True)
    env = gym.make("MountainCarContinuous-v0")
    t0 = time.time()
    agent = DynaQAgent(n_pos_bins=20, n_vel_bins=20, n_actions=15,
                       n_planning_steps=n, q_init=0.0)
    agent.train_agent(env, **TRAIN_KWARGS)
    elapsed = time.time() - t0

    res = agent.test_agent(env, episodes=EVAL_EPISODES)
    env.close()

    ci95 = 1.96 * res["std_reward"] / (EVAL_EPISODES ** 0.5)

    # Primer éxito (reward > 0 suele indicar éxito)
    rewards = np.array(agent.training_rewards)
    success_eps = np.where(rewards > 0)[0]
    first_success = int(success_eps[0] + 1) if len(success_eps) > 0 else None

    results.append({
        "n_planning": n,
        "mean_reward": res["mean_reward"],
        "std_reward": res["std_reward"],
        "success_rate": res["success_rate"],
        "ci95": ci95,
        "first_success_ep": first_success,
        "training_time_s": elapsed,
    })
    all_rewards[n] = rewards
    print(f"mean={res['mean_reward']:.2f} success={res['success_rate']:.0%} "
          f"first_ep={first_success} time={elapsed:.0f}s")

df = pd.DataFrame(results)
df.to_csv(str(MODELS / "dyna_planning_sweep.csv"), index=False)

print("\n=== TABLA RESUMEN ===")
print(df.to_string(index=False))

# ---------- Figura 1: Curvas de aprendizaje ----------
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

WINDOW = 50
for n in N_PLANNING_VALUES:
    r = all_rewards[n]
    rm = np.convolve(r, np.ones(WINDOW) / WINDOW, mode="valid")
    axes[0].plot(np.arange(WINDOW - 1, len(r)), rm,
                 color=colors[n], lw=2 if n in [0, 10, 50] else 1.2,
                 label=f"n={n}" + (" (QL puro)" if n == 0 else ""),
                 alpha=0.9)

axes[0].axhline(0, color="gray", ls="--", alpha=0.4)
axes[0].set_xlabel("Episodio")
axes[0].set_ylabel(f"Media móvil recompensa (ventana={WINDOW})")
axes[0].set_title("Dyna-Q: Impacto de n_planning en Aprendizaje")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# ---------- Figura 2: Métricas finales vs n ----------
ax2 = axes[1]
ax2_r = ax2.twinx()

ns = df["n_planning"].values
ax2.bar(ns, df["success_rate"] * 100, width=2.5, alpha=0.5,
        color=[colors[n] for n in ns], label="Éxito (%)")
ax2_r.errorbar(ns, df["mean_reward"], yerr=df["ci95"],
               color="navy", marker="o", lw=2, capsize=4, label="Recompensa media (IC95)")

ax2.set_xlabel("n_planning_steps")
ax2.set_ylabel("Tasa de éxito (%)", color="gray")
ax2_r.set_ylabel("Recompensa media ± IC95", color="navy")
ax2.set_title("Dyna-Q: Rendimiento Final vs n_planning")
ax2.set_ylim(0, 115)
ax2.set_xticks(ns)
ax2.legend(loc="upper left")
ax2_r.legend(loc="upper right")
ax2.grid(alpha=0.3)

fig.suptitle(f"Experimento Dyna-Q n_planning — {EPISODES} episodios, α=0.2, γ=0.99, decay=0.9995",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(str(FIGURES / "dyna_planning_sweep.png"), dpi=150, bbox_inches="tight")
print(f"\nFigura: reports/figures/dyna_planning_sweep.png")

# Análisis del trade-off
print("\n=== ANÁLISIS TRADE-OFF ===")
print(f"{'n':>4}  {'Éxito':>7}  {'Reward':>8}  {'1er ep':>8}  {'Tiempo':>8}")
for _, row in df.iterrows():
    n = int(row["n_planning"])
    fse = row["first_success_ep"]
    fe = f"ep {int(fse)}" if (fse == fse and fse is not None) else "nunca"
    print(f"{n:>4}  {row['success_rate']:>7.0%}  {row['mean_reward']:>8.2f}  "
          f"{fe:>8}  {row['training_time_s']:>6.0f}s")
