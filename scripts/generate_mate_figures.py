#!/usr/bin/env python3
"""
Genera figuras del experimento riguroso de MATE y corre experimentos complementarios.

Experimentos adicionales (antes no registrados):
  1. Agente final (mob_only d4) vs Random        → confirmar dominancia del agente final
  2. Mirror match: mob_only d4 vs mob_only d4     → cuantificar ventaja del P1 a depth=4

Figuras generadas:
  reports/figures/rigorous_depth_sweep.png      — sweep depth × heurística (vs baseline d3)
  reports/figures/rigorous_roundrobin.png        — round-robin con Wilson CIs y p-valores
  reports/figures/rigorous_minimax_vs_exp.png    — Minimax vs Expectimax con CIs
  reports/figures/rigorous_vs_stratagem.png      — agente final vs Stratagem (depth=4)

Datos adicionales:
  models/mate/rigorous_extra.csv                 — final vs Random + mirror match

Ejecución (desde Isolation/):
    python ../scripts/generate_mate_figures.py
"""

import csv, sys, math, time, random as pyrandom
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Isolation"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from isolation_env import IsolationEnv
from minimax_agent import MinimaxAgent
from random_agent import RandomAgent
from heuristics import eval_mobility_only

try:
    from stratagem import Stratagem
    HAS_STRATAGEM = True
except Exception:
    HAS_STRATAGEM = False

FIGURES = Path(__file__).parent.parent / "reports" / "figures"
MODELS  = Path(__file__).parent.parent / "models" / "mate"
FIGURES.mkdir(parents=True, exist_ok=True)

STYLE = {
    "sig":    "#2196F3",   # azul — significativo
    "ns":     "#B0BEC5",   # gris — no significativo
    "win":    "#43A047",   # verde — gana
    "lose":   "#E53935",   # rojo — pierde
    "draw":   "#FB8C00",   # naranja — empate/n.s.
    "strat":  "#6A1B9A",   # violeta — vs Stratagem
}

# ── Utilidades ─────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def normal_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))

def binom_p(wins, n, p0=0.5):
    if n == 0:
        return 1.0
    se = math.sqrt(n * p0 * (1 - p0))
    z  = (wins - n * p0) / se
    return 2.0 * (1.0 - normal_cdf(abs(z)))

def play_one(f1, f2):
    a1, a2 = f1(1), f2(2)
    env = IsolationEnv()
    board = env.reset()
    done = False
    winner = 0
    while not done:
        cp = env.current_player
        action = a1.next_action(board) if cp == 1 else a2.next_action(board)
        if action is None:
            winner = cp % 2 + 1
            break
        board, _, done, winner, _ = env.step(action)
    return winner

def balanced(fa, fb, n):
    assert n % 2 == 0
    half = n // 2
    wins_a = 0
    for _ in range(half):
        if play_one(fa, fb) == 1:
            wins_a += 1
    for _ in range(half):
        if play_one(fb, fa) == 2:
            wins_a += 1
    wins_b = n - wins_a
    p = binom_p(wins_a, n)
    ci_lo, ci_hi = wilson_ci(wins_a, n)
    return {"wins_a": wins_a, "wins_b": wins_b, "n": n,
            "wr_a": wins_a/n, "p": p, "ci_lo": ci_lo, "ci_hi": ci_hi}

def mm4():
    return lambda p: MinimaxAgent(p, max_depth=4,
                                  heuristic=eval_mobility_only, use_alpha_beta=True)

# ── Experimentos adicionales ───────────────────────────────────────────────

def run_extra_experiments():
    print("\n" + "="*60)
    print("EXPERIMENTOS COMPLEMENTARIOS")
    print("="*60)

    extra_rows = []

    # 1. Agente final vs Random
    print("\n[1/2] mob_only d4 vs Random (100 partidas balanceadas)...")
    t0 = time.time()
    res = balanced(mm4(), lambda p: RandomAgent(p), 100)
    t = time.time() - t0
    print(f"  → {res['wins_a']}/100  wr={res['wr_a']:.3f}  "
          f"CI=[{res['ci_lo']:.3f},{res['ci_hi']:.3f}]  "
          f"p={res['p']:.5f}  ({t:.0f}s)")
    extra_rows.append({
        "experiment": "final_agent_vs_random",
        "agent_a": "mob_only_d4",
        "agent_b": "random",
        "wins_a": res["wins_a"], "n": res["n"],
        "win_rate_a": round(res["wr_a"], 4),
        "p_value": round(res["p"], 6),
        "ci_lo": round(res["ci_lo"], 4),
        "ci_hi": round(res["ci_hi"], 4),
    })

    # 2. Mirror match (P1 advantage)
    print("\n[2/2] Mirror match: mob_only d4 vs mob_only d4 (100 partidas)...")
    print("      (mide ventaja del primer jugador a depth=4)")
    t0 = time.time()
    p1_wins = 0
    for _ in range(100):
        a1 = MinimaxAgent(1, 4, eval_mobility_only, True)
        a2 = MinimaxAgent(2, 4, eval_mobility_only, True)
        env = IsolationEnv()
        board = env.reset()
        done = False
        winner = 0
        while not done:
            cp = env.current_player
            action = a1.next_action(board) if cp == 1 else a2.next_action(board)
            if action is None:
                winner = cp % 2 + 1
                break
            board, _, done, winner, _ = env.step(action)
        if winner == 1:
            p1_wins += 1
    p_mirror = binom_p(p1_wins, 100)
    ci_lo_m, ci_hi_m = wilson_ci(p1_wins, 100)
    t = time.time() - t0
    print(f"  → P1 gana {p1_wins}/100  wr={p1_wins/100:.3f}  "
          f"CI=[{ci_lo_m:.3f},{ci_hi_m:.3f}]  p={p_mirror:.5f}  ({t:.0f}s)")
    extra_rows.append({
        "experiment": "mirror_match_p1_advantage",
        "agent_a": "mob_only_d4_as_P1",
        "agent_b": "mob_only_d4_as_P2",
        "wins_a": p1_wins, "n": 100,
        "win_rate_a": round(p1_wins/100, 4),
        "p_value": round(p_mirror, 6),
        "ci_lo": round(ci_lo_m, 4),
        "ci_hi": round(ci_hi_m, 4),
    })

    out = MODELS / "rigorous_extra.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=extra_rows[0].keys())
        w.writeheader()
        w.writerows(extra_rows)
    print(f"\n  → Guardado: {out.name}")
    return extra_rows

# ── Figura 1: Depth Sweep ──────────────────────────────────────────────────

def fig_depth_sweep():
    rows = read_csv(MODELS / "rigorous_depth_sweep.csv")

    labels, wrs, ci_los, ci_his, colors, pvals = [], [], [], [], [], []
    for r in rows:
        h = r["heuristic"].replace("mob_only", "mob").replace("mob_territory", "mob+terr")
        labels.append(f"{h}\nd={r['depth']}")
        wr = float(r["win_rate"])
        wrs.append(wr)
        lo, hi = wilson_ci(int(r["wins_vs_baseline"]), int(r["n"]))
        ci_los.append(wr - lo)
        ci_his.append(hi - wr)
        p = float(r["p_value"])
        pvals.append(p)
        if p < 0.01:
            colors.append(STYLE["sig"] if wr > 0.5 else STYLE["lose"])
        elif p < 0.05:
            colors.append("#90CAF9" if wr > 0.5 else "#EF9A9A")
        else:
            colors.append(STYLE["ns"])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, wrs, color=colors, width=0.65,
                  yerr=[ci_los, ci_his], capsize=4,
                  error_kw={"elinewidth": 1.2, "ecolor": "black", "alpha": 0.6})

    ax.axhline(0.5, color="black", lw=1.2, ls="--", alpha=0.5, label="Baseline (mob_only d3)")
    ax.axvline(2.5, color="gray", lw=0.8, ls=":")
    ax.axvline(5.5, color="gray", lw=0.8, ls=":")
    ax.text(1.0,  0.97, "depth=2", ha="center", va="top", transform=ax.get_xaxis_transform(),
            fontsize=9, color="gray")
    ax.text(4.0,  0.97, "depth=3", ha="center", va="top", transform=ax.get_xaxis_transform(),
            fontsize=9, color="gray")
    ax.text(7.0,  0.97, "depth=4", ha="center", va="top", transform=ax.get_xaxis_transform(),
            fontsize=9, color="gray")

    # Significance markers
    for i, (wr, p) in enumerate(zip(wrs, pvals)):
        if p < 0.01:
            ax.text(i, wr + ci_his[i] + 0.025, "**", ha="center", fontsize=12, fontweight="bold",
                    color=STYLE["sig"] if wr > 0.5 else STYLE["lose"])
        elif p < 0.05:
            ax.text(i, wr + ci_his[i] + 0.025, "*", ha="center", fontsize=12)
        else:
            ax.text(i, wr + ci_his[i] + 0.02, "n.s.", ha="center", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.15, 1.05)
    ax.set_ylabel("Win rate vs mob_only d3")
    ax.set_title("Sweep de Profundidad × Heurística vs Baseline (mob_only depth=3)\n"
                 "100 partidas balanceadas por celda — ** p<0.01, * p<0.05, n.s. p≥0.05")

    legend_elements = [
        mpatches.Patch(color=STYLE["sig"],  label="Significativo (mejor, p<0.01)"),
        mpatches.Patch(color=STYLE["lose"], label="Significativo (peor, p<0.01)"),
        mpatches.Patch(color=STYLE["ns"],   label="No significativo"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURES / "rigorous_depth_sweep.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out.name}")

# ── Figura 2: Round-Robin ──────────────────────────────────────────────────

def fig_roundrobin():
    summary = read_csv(MODELS / "rigorous_summary.csv")
    summary.sort(key=lambda r: float(r["win_rate"]), reverse=True)

    names = [r["heuristic"] for r in summary]
    wrs   = [float(r["win_rate"]) for r in summary]
    ci_lo = [float(r["ci_lo"]) for r in summary]
    ci_hi = [float(r["ci_hi"]) for r in summary]
    err_lo = [wr - lo for wr, lo in zip(wrs, ci_lo)]
    err_hi = [hi - wr for wr, hi in zip(wrs, ci_hi)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart with CIs
    x = np.arange(len(names))
    ax1.bar(x, wrs, color=STYLE["ns"], width=0.6,
            yerr=[err_lo, err_hi], capsize=5,
            error_kw={"elinewidth": 1.5, "ecolor": "black"})
    ax1.axhline(0.5, color="black", lw=1.2, ls="--", alpha=0.6, label="50% (sin diferencia)")
    for i, wr in enumerate(wrs):
        ax1.text(i, wr + err_hi[i] + 0.015, "n.s.", ha="center", fontsize=9, color="gray")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax1.set_ylim(0.35, 0.70)
    ax1.set_ylabel("Win rate (round-robin)")
    ax1.set_title("Win rate global — Round-Robin Riguroso\n"
                  "(100 partidas/par, Bonferroni α=0.005: ninguna diferencia significativa)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Matchup matrix heatmap
    matrix = read_csv(MODELS / "rigorous_matchup_matrix.csv")
    hnames = [r["heuristic"] for r in matrix]
    mat = np.zeros((len(hnames), len(hnames)))
    for i, r in enumerate(matrix):
        for j, h in enumerate(hnames):
            val = r.get(f"vs_{h}", "0")
            mat[i][j] = int(val) if val and val != "0" else 0

    # Normalize to win rate
    mat_wr = np.zeros_like(mat, dtype=float)
    for i in range(len(hnames)):
        for j in range(len(hnames)):
            if i != j:
                total = mat[i][j] + mat[j][i]
                mat_wr[i][j] = mat[i][j] / total if total > 0 else 0.5
            else:
                mat_wr[i][j] = np.nan

    im = ax2.imshow(mat_wr, cmap="RdYlGn", vmin=0.35, vmax=0.65, aspect="auto")
    ax2.set_xticks(range(len(hnames)))
    ax2.set_yticks(range(len(hnames)))
    ax2.set_xticklabels(hnames, rotation=30, ha="right", fontsize=8)
    ax2.set_yticklabels(hnames, fontsize=8)
    ax2.set_title("Matriz de matchups\n(win rate fila vs columna)")
    for i in range(len(hnames)):
        for j in range(len(hnames)):
            if i != j:
                ax2.text(j, i, f"{mat_wr[i,j]:.2f}", ha="center", va="center",
                         fontsize=7, color="black")
            else:
                ax2.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.suptitle("Round-Robin Riguroso: 5 Heurísticas — Todas Equivalentes (depth=3)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = FIGURES / "rigorous_roundrobin.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out.name}")

# ── Figura 3: Minimax vs Expectimax ───────────────────────────────────────

def fig_minimax_vs_exp():
    rows = read_csv(MODELS / "rigorous_minimax_vs_exp.csv")

    labels = [f"{r['agent_a']}\nvs {r['agent_b']}" for r in rows]
    wrs    = [float(r["win_rate_a"]) for r in rows]
    pvals  = [float(r["p_value"]) for r in rows]
    ci_lo  = [float(r["ci_lo"]) for r in rows]
    ci_hi  = [float(r["ci_hi"]) for r in rows]
    err_lo = [wr - lo for wr, lo in zip(wrs, ci_lo)]
    err_hi = [hi - wr for wr, hi in zip(wrs, ci_hi)]

    colors = []
    for wr, p in zip(wrs, pvals):
        if p >= 0.05:
            colors.append(STYLE["draw"])
        elif wr > 0.5:
            colors.append(STYLE["win"])
        else:
            colors.append(STYLE["lose"])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    ax.bar(x, wrs, color=colors, width=0.55,
           yerr=[err_lo, err_hi], capsize=6,
           error_kw={"elinewidth": 1.5, "ecolor": "black"})
    ax.axhline(0.5, color="black", lw=1.2, ls="--", alpha=0.6, label="50% (sin diferencia)")

    for i, (wr, p) in enumerate(zip(wrs, pvals)):
        offset = err_hi[i] + 0.03
        if p < 0.001:
            label = f"p≈0\n***"
        elif p < 0.05:
            label = f"p={p:.3f}\n*"
        else:
            label = f"p={p:.3f}\nn.s."
        ax.text(i, wr + offset, label, ha="center", fontsize=8,
                color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.20, 1.10)
    ax.set_ylabel("Win rate de Minimax")
    ax.set_title("Minimax vs Expectimax — 100 partidas balanceadas por matchup\n"
                 "Heurística: mob_only | Barra = win rate de Minimax + IC Wilson 95%")

    legend_elements = [
        mpatches.Patch(color=STYLE["win"],  label="Minimax gana (p<0.05)"),
        mpatches.Patch(color=STYLE["lose"], label="Expectimax gana (p<0.05)"),
        mpatches.Patch(color=STYLE["draw"], label="Sin diferencia significativa"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURES / "rigorous_minimax_vs_exp.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out.name}")

# ── Figura 4: vs Stratagem ────────────────────────────────────────────────

def fig_vs_stratagem(extra_rows=None):
    strat = read_csv(MODELS / "rigorous_vs_stratagem.csv")[0]
    wr    = float(strat["win_rate"])
    ci_lo = float(strat["ci_lo"])
    ci_hi = float(strat["ci_hi"])

    fig, ax = plt.subplots(figsize=(8, 4))

    agents = ["Minimax-AB\nmob_only d4\nvs Stratagem"]
    wrs    = [wr]
    err_lo = [wr - ci_lo]
    err_hi = [ci_hi - wr]
    colors = [STYLE["strat"]]

    if extra_rows:
        # Add final vs Random result
        for r in extra_rows:
            if r["experiment"] == "final_agent_vs_random":
                wr_r = float(r["win_rate_a"])
                lo_r, hi_r = float(r["ci_lo"]), float(r["ci_hi"])
                agents.append("Minimax-AB\nmob_only d4\nvs Random")
                wrs.append(wr_r)
                err_lo.append(wr_r - lo_r)
                err_hi.append(hi_r - wr_r)
                colors.append(STYLE["sig"])

    x = np.arange(len(agents))
    ax.bar(x, wrs, color=colors, width=0.45,
           yerr=[err_lo, err_hi], capsize=7,
           error_kw={"elinewidth": 1.8, "ecolor": "black"})
    ax.axhline(0.5, color="black", lw=1.2, ls="--", alpha=0.6, label="50%")
    ax.axhline(1.0, color="gray", lw=0.5, ls=":", alpha=0.5)

    for i, (w, lo, hi) in enumerate(zip(wrs, err_lo, err_hi)):
        ax.text(i, w + hi + 0.025, f"{w:.0%}", ha="center", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylim(0.3, 1.15)
    ax.set_ylabel("Win rate (partidas balanceadas)")
    ax.set_title("Agente Final: Minimax-AB mob_only depth=4\n"
                 "100 partidas balanceadas — IC Wilson 95%")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURES / "rigorous_vs_stratagem.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out.name}")

# ── Figura 5: Resumen ejecutivo (tabla visual) ─────────────────────────────

def fig_summary(extra_rows=None):
    """Un único gráfico de resumen que muestra los hallazgos principales de MATE."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("PROYECTO MATE — Resumen Experimental\n"
                 "MinimaxAgent con Alpha-Beta Pruning | Isolation 4×4",
                 fontsize=12, fontweight="bold")

    # Panel 1: AB Impact (nodos)
    ax = axes[0]
    ab = read_csv(MODELS / "ab_impact_results.csv")
    depths = sorted(set(int(r["depth"]) for r in ab))
    configs = ["Pure Minimax", "AB Pruning", "AB + Move Ordering"]
    cfg_colors = ["#EF5350", "#42A5F5", "#66BB6A"]
    x = np.arange(len(depths))
    width = 0.25
    for i, (cfg, col) in enumerate(zip(configs, cfg_colors)):
        vals = [int(r["avg_nodes"]) for r in ab if r["label"] == cfg]
        if vals:
            bars = ax.bar(x + i*width, vals, width, label=cfg, color=col, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"depth={d}" for d in depths])
    ax.set_ylabel("Nodos expandidos (log)")
    ax.set_title("Impacto de Alpha-Beta Pruning\n(nodos promedio por partida)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Depth sweep — solo los significativos
    ax = axes[1]
    ds = read_csv(MODELS / "rigorous_depth_sweep.csv")
    key_configs = [("mob_only", 2), ("mob_only", 3), ("mob_only", 4),
                   ("mob_territory", 4), ("full", 4)]
    xlabels, vals, errs_lo, errs_hi, bcols = [], [], [], [], []
    for h, d in key_configs:
        r = next((x for x in ds if x["heuristic"] == h and int(x["depth"]) == d), None)
        if r:
            wr = float(r["win_rate"])
            lo, hi = wilson_ci(int(r["wins_vs_baseline"]), int(r["n"]))
            xlabels.append(f"{h[:3]}+\nd={d}" if h != "mob_only" else f"mob\nd={d}")
            vals.append(wr)
            errs_lo.append(wr - lo)
            errs_hi.append(hi - wr)
            p = float(r["p_value"])
            if p < 0.01 and wr > 0.5:
                bcols.append(STYLE["sig"])
            elif p < 0.01 and wr < 0.5:
                bcols.append(STYLE["lose"])
            else:
                bcols.append(STYLE["ns"])
    xpos = np.arange(len(xlabels))
    ax.bar(xpos, vals, color=bcols, width=0.55,
           yerr=[errs_lo, errs_hi], capsize=4,
           error_kw={"elinewidth": 1.2, "ecolor": "black"})
    ax.axhline(0.5, color="black", lw=1.2, ls="--", alpha=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylim(0.2, 0.85)
    ax.set_ylabel("Win rate vs mob_only d3")
    ax.set_title("Profundidad vs Heurística\n(** p<0.01 vs baseline mob_only d3)")
    for i, (v, lo, hi, p) in enumerate(zip(vals, errs_lo, errs_hi,
            [float(next(x for x in ds if x["heuristic"]==h and int(x["depth"])==d)["p_value"])
             for h, d in key_configs])):
        if p < 0.01:
            ax.text(i, v + hi + 0.025, "**", ha="center", fontsize=11, fontweight="bold",
                    color=STYLE["sig"] if v > 0.5 else STYLE["lose"])
        else:
            ax.text(i, v + hi + 0.015, "n.s.", ha="center", fontsize=7, color="gray")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Benchmark final
    ax = axes[2]
    labels_b, wrs_b, errs_lo_b, errs_hi_b, cols_b = [], [], [], [], []

    # MM vs EX d3
    mv = read_csv(MODELS / "rigorous_minimax_vs_exp.csv")
    r_d3 = next(r for r in mv if r["agent_b"] == "EX_d3")
    wr_d3 = float(r_d3["win_rate_a"])
    lo, hi = float(r_d3["ci_lo"]), float(r_d3["ci_hi"])
    labels_b.append("MM d3\nvs EX d3")
    wrs_b.append(wr_d3); errs_lo_b.append(wr_d3-lo); errs_hi_b.append(hi-wr_d3)
    cols_b.append(STYLE["win"])

    # vs Stratagem
    sr = read_csv(MODELS / "rigorous_vs_stratagem.csv")[0]
    wr_s = float(sr["win_rate"])
    lo, hi = float(sr["ci_lo"]), float(sr["ci_hi"])
    labels_b.append("mob_only d4\nvs Stratagem")
    wrs_b.append(wr_s); errs_lo_b.append(wr_s-lo); errs_hi_b.append(hi-wr_s)
    cols_b.append(STYLE["strat"])

    # vs Random (if available)
    if extra_rows:
        for r in extra_rows:
            if r["experiment"] == "final_agent_vs_random":
                wr_r = float(r["win_rate_a"])
                lo_r, hi_r = float(r["ci_lo"]), float(r["ci_hi"])
                labels_b.append("mob_only d4\nvs Random")
                wrs_b.append(wr_r)
                errs_lo_b.append(wr_r - lo_r)
                errs_hi_b.append(hi_r - wr_r)
                cols_b.append(STYLE["sig"])

    xpos = np.arange(len(labels_b))
    ax.bar(xpos, wrs_b, color=cols_b, width=0.5,
           yerr=[errs_lo_b, errs_hi_b], capsize=5,
           error_kw={"elinewidth": 1.5, "ecolor": "black"})
    ax.axhline(0.5, color="black", lw=1.2, ls="--", alpha=0.6)
    for i, w in enumerate(wrs_b):
        ax.text(i, w + errs_hi_b[i] + 0.025, f"{w:.0%}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels_b, fontsize=8)
    ax.set_ylim(0.3, 1.15)
    ax.set_ylabel("Win rate")
    ax.set_title("Benchmarks del agente final\n(mob_only depth=4, 100 partidas c/u)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES / "rigorous_summary.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("GENERACIÓN DE FIGURAS — PROYECTO MATE")
    print("="*60)

    # Experimentos complementarios
    extra_rows = run_extra_experiments()

    # Figuras
    print("\n" + "="*60)
    print("GENERANDO FIGURAS")
    print("="*60)
    fig_depth_sweep()
    fig_roundrobin()
    fig_minimax_vs_exp()
    fig_vs_stratagem(extra_rows)
    fig_summary(extra_rows)

    print("\n✓ Completado. Figuras guardadas en reports/figures/")
    print("  rigorous_depth_sweep.png")
    print("  rigorous_roundrobin.png")
    print("  rigorous_minimax_vs_exp.png")
    print("  rigorous_vs_stratagem.png")
    print("  rigorous_summary.png")
    print("  models/mate/rigorous_extra.csv")


if __name__ == "__main__":
    main()
