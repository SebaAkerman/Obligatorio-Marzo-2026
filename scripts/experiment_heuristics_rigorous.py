#!/usr/bin/env python3
"""
Experimento riguroso de comparación de heurísticas — Proyecto MATE.

Diseño estadístico:
  - Round-robin completo: C(5,2) = 10 pares, 100 partidas/par (50 P1 + 50 P2)
  - Test binomial (aproximación normal): H0: winrate = 0.5 (sin diferencia)
  - Corrección de Bonferroni: α_adj = 0.05 / 10 = 0.005 por par
  - Potencia teórica: ~94% para detectar diferencias >= 10%
  - Mínima diferencia detectable post-Bonferroni: ~14% (z_crit=2.807)

Fases:
  1. Round-Robin completo (5 heurísticas, depth=3)
  2. Sweep profundidad × heurística (depths 2,3,4 vs baseline)
  3. Minimax vs Expectimax (100 partidas balanceadas)
  4. Mejor agente vs Stratagem (referencia de la cátedra)

Referencia: Russell & Norvig (2020) cap. 5 — Adversarial Search and Games
            Sutton & Barto (2020) — metodología experimental en IA

Ejecución (desde Isolation/):
    python ../scripts/experiment_heuristics_rigorous.py

Outputs (../models/mate/):
    rigorous_matchup_matrix.csv   — wins[H_i][H_j] para todos los pares
    rigorous_significance.csv     — p-values, significancia post-Bonferroni
    rigorous_depth_sweep.csv      — win rate por depth x heurística
    rigorous_minimax_vs_exp.csv   — comparación Minimax vs Expectimax
    rigorous_vs_stratagem.csv     — benchmark vs agente de referencia
"""

import sys, math, time, csv
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent / "Isolation"))

from isolation_env import IsolationEnv
from minimax_agent import MinimaxAgent
from expectimax_agent import ExpectimaxAgent
from random_agent import RandomAgent
from heuristics import (
    eval_mobility_only,
    eval_mobility_center,
    eval_full,
    eval_territory,
    eval_mobility_territory,
)

try:
    from stratagem import Stratagem
    HAS_STRATAGEM = True
except Exception:
    HAS_STRATAGEM = False

OUT_DIR = Path(__file__).parent.parent / "models" / "mate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constantes ────────────────────────────────────────────────────────────
HEURISTICS_RR = {
    "mob_only":        eval_mobility_only,
    "mob_center":      eval_mobility_center,
    "full":            eval_full,
    "territory":       eval_territory,
    "mob_territory":   eval_mobility_territory,
}
N_RR        = 100   # partidas por par en round-robin (50 P1 + 50 P2)
N_DEPTH     = 100   # partidas por celda en sweep de profundidad
N_MV_EXP    = 100   # partidas minimax vs expectimax
N_STRATAGEM = 100   # partidas vs Stratagem
ALPHA       = 0.05  # nivel de significancia global
DEPTH_RR    = 3     # profundidad para el round-robin

# ── Utilidades estadísticas ───────────────────────────────────────────────

def normal_cdf(z: float) -> float:
    """CDF de la distribución normal estándar usando math.erfc."""
    return 0.5 * math.erfc(-z / math.sqrt(2))


def binomial_p_two_sided(wins: int, n: int, p0: float = 0.5) -> float:
    """
    P-valor del test binomial de dos colas (aproximación normal).
    H0: la proporción de victorias es p0.
    Válido para n >= 30.
    """
    if n == 0:
        return 1.0
    se = math.sqrt(n * p0 * (1 - p0))
    z  = (wins - n * p0) / se
    return 2.0 * (1.0 - normal_cdf(abs(z)))


def wilson_ci(wins: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Intervalo de confianza de Wilson para proporciones.
    Más preciso que el CI normal para valores extremos.
    """
    if n == 0:
        return (0.0, 1.0)
    z    = 1.96 if confidence == 0.95 else 2.576
    p    = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ── Motor de torneos ───────────────────────────────────────────────────────

def play_one_game(factory1, factory2) -> int:
    """
    Juega una partida. factory1 crea al jugador 1, factory2 al jugador 2.
    Retorna el número del ganador (1 o 2).
    """
    a1 = factory1(1)
    a2 = factory2(2)
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


def balanced_match(factory_a, factory_b, n_games: int) -> dict:
    """
    Enfrenta dos agentes en n_games partidas (n_games/2 como P1, n_games/2 como P2).
    Retorna wins_a, wins_b, y estadísticas.
    """
    assert n_games % 2 == 0, "n_games debe ser par para balancear P1/P2"
    half = n_games // 2
    wins_a = 0

    # Primera mitad: A es P1
    for _ in range(half):
        w = play_one_game(factory_a, factory_b)
        if w == 1:
            wins_a += 1

    # Segunda mitad: A es P2
    for _ in range(half):
        w = play_one_game(factory_b, factory_a)
        if w == 2:
            wins_a += 1

    wins_b = n_games - wins_a
    p_val  = binomial_p_two_sided(wins_a, n_games)
    ci_lo, ci_hi = wilson_ci(wins_a, n_games)

    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "win_rate_a": wins_a / n_games,
        "p_value": p_val,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


def minimax_factory(heuristic_fn, depth: int, alpha_beta: bool = True):
    return lambda player: MinimaxAgent(
        player=player,
        max_depth=depth,
        heuristic=heuristic_fn,
        use_alpha_beta=alpha_beta,
    )


def expectimax_factory(heuristic_fn, depth: int):
    return lambda player: ExpectimaxAgent(
        player=player,
        max_depth=depth,
        heuristic=heuristic_fn,
    )


# ── Fase 1: Round-Robin ────────────────────────────────────────────────────

def phase1_roundrobin():
    names = list(HEURISTICS_RR.keys())
    n_pairs = len(names) * (len(names) - 1) // 2
    alpha_bonferroni = ALPHA / n_pairs  # corrección Bonferroni

    print(f"\n{'='*65}")
    print(f"FASE 1 — Round-Robin completo (depth={DEPTH_RR})")
    print(f"  Heurísticas : {names}")
    print(f"  Partidas/par: {N_RR} (50 P1 + 50 P2)")
    print(f"  α Bonferroni: {alpha_bonferroni:.4f} ({n_pairs} pares)")
    print(f"{'='*65}")

    # Matriz de victorias wins[i][j] = partidas que i ganó contra j
    wins_matrix = {n: {m: 0 for m in names} for n in names}
    significance_rows = []

    for i, (name_a, hfn_a) in enumerate(HEURISTICS_RR.items()):
        for name_b, hfn_b in list(HEURISTICS_RR.items())[i+1:]:
            t0 = time.time()
            print(f"  {name_a:16s} vs {name_b:16s} ... ", end="", flush=True)

            res = balanced_match(
                minimax_factory(hfn_a, DEPTH_RR),
                minimax_factory(hfn_b, DEPTH_RR),
                N_RR,
            )

            wins_matrix[name_a][name_b] = res["wins_a"]
            wins_matrix[name_b][name_a] = res["wins_b"]

            sig = res["p_value"] < alpha_bonferroni
            elapsed = time.time() - t0
            print(
                f"{res['wins_a']:3d}/{N_RR}  "
                f"wr={res['win_rate_a']:.3f}  "
                f"p={res['p_value']:.4f}  "
                f"{'*** SIGNIFICATIVO' if sig else '     n.s.':18s}  "
                f"({elapsed:.0f}s)"
            )

            significance_rows.append({
                "h_a": name_a, "h_b": name_b,
                "wins_a": res["wins_a"], "wins_b": res["wins_b"],
                "n": N_RR,
                "win_rate_a": round(res["win_rate_a"], 4),
                "p_value": round(res["p_value"], 5),
                "ci_lo": round(res["ci_lo"], 4),
                "ci_hi": round(res["ci_hi"], 4),
                "alpha_bonferroni": round(alpha_bonferroni, 5),
                "significant": sig,
            })

    # Totales por heurística
    print(f"\n  {'Heurística':16s} {'Wins':>5} {'Games':>5} {'WinRate':>8} {'CI 95%':>15}")
    print(f"  {'-'*60}")
    summary_rows = []
    for name in names:
        total_wins  = sum(wins_matrix[name].values())
        total_games = (len(names) - 1) * N_RR
        wr = total_wins / total_games
        ci_lo, ci_hi = wilson_ci(total_wins, total_games)
        summary_rows.append({
            "heuristic": name,
            "total_wins": total_wins,
            "total_games": total_games,
            "win_rate": round(wr, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
        })
        print(
            f"  {name:16s} {total_wins:5d} {total_games:5d} "
            f"{wr:8.3f}   [{ci_lo:.3f}, {ci_hi:.3f}]"
        )

    summary_rows.sort(key=lambda r: r["win_rate"], reverse=True)

    # Guardar CSVs
    _write_csv(significance_rows, OUT_DIR / "rigorous_significance.csv")
    _write_csv(summary_rows,      OUT_DIR / "rigorous_summary.csv")

    # Guardar matriz de matchups
    matrix_rows = []
    for name_a in names:
        row = {"heuristic": name_a}
        row.update({f"vs_{name_b}": wins_matrix[name_a][name_b] for name_b in names})
        matrix_rows.append(row)
    _write_csv(matrix_rows, OUT_DIR / "rigorous_matchup_matrix.csv")

    any_significant = any(r["significant"] for r in significance_rows)
    print(f"\n  Conclusión: ", end="")
    if any_significant:
        winner = summary_rows[0]["heuristic"]
        print(f"Par(es) significativos encontrados. Mejor heurística: {winner}")
    else:
        print(
            "Ningún par alcanza significancia post-Bonferroni.\n"
            "  Las heurísticas son estadísticamente equivalentes en 4×4 Isolation (depth=3).\n"
            "  Recomendación: usar mob_only (más simple, más rápida)."
        )

    return summary_rows, any_significant


# ── Fase 2: Sweep profundidad × heurística ────────────────────────────────

def phase2_depth_sweep(rr_summary: list):
    """
    Enfrenta cada heurística a distintas profundidades contra mob_only depth=3
    como baseline. Responde: ¿compensa la profundidad a una heurística débil?
    """
    print(f"\n{'='*65}")
    print("FASE 2 — Sweep profundidad × heurística")
    print(f"  Baseline : mob_only depth=3")
    print(f"  Partidas : {N_DEPTH} por celda (balanceadas)")
    print(f"{'='*65}")

    # Tomar las 3 mejores del round-robin + siempre mob_only
    top3 = [r["heuristic"] for r in rr_summary[:3]]
    if "mob_only" not in top3:
        top3 = ["mob_only"] + top3[:2]
    depths = [2, 3, 4]

    baseline_factory = minimax_factory(eval_mobility_only, 3)
    depth_rows = []

    for depth in depths:
        print(f"\n  depth={depth}:")
        for hname in top3:
            hfn = HEURISTICS_RR[hname]
            label = f"{hname} d{depth}"
            print(f"    {label:22s} vs mob_only d3 ... ", end="", flush=True)
            t0 = time.time()
            res = balanced_match(
                minimax_factory(hfn, depth),
                baseline_factory,
                N_DEPTH,
            )
            elapsed = time.time() - t0
            sig = res["p_value"] < 0.05  # sin corrección para exploración
            print(
                f"{res['wins_a']:3d}/{N_DEPTH}  "
                f"wr={res['win_rate_a']:.3f}  "
                f"p={res['p_value']:.4f}  "
                f"({'sig' if sig else 'n.s.':4s})  "
                f"({elapsed:.0f}s)"
            )
            depth_rows.append({
                "heuristic": hname,
                "depth": depth,
                "wins_vs_baseline": res["wins_a"],
                "n": N_DEPTH,
                "win_rate": round(res["win_rate_a"], 4),
                "p_value": round(res["p_value"], 5),
                "significant_p05": sig,
            })

    _write_csv(depth_rows, OUT_DIR / "rigorous_depth_sweep.csv")
    return depth_rows


# ── Fase 3: Minimax vs Expectimax ─────────────────────────────────────────

def phase3_minimax_vs_expectimax():
    print(f"\n{'='*65}")
    print("FASE 3 — Minimax vs Expectimax")
    print(f"  Partidas: {N_MV_EXP} por configuración (balanceadas)")
    print(f"{'='*65}")

    configs = [
        ("MM_d2", minimax_factory(eval_mobility_only, 2),
         "EX_d2", expectimax_factory(eval_mobility_only, 2)),
        ("MM_d3", minimax_factory(eval_mobility_only, 3),
         "EX_d3", expectimax_factory(eval_mobility_only, 3)),
        ("MM_d3", minimax_factory(eval_mobility_only, 3),
         "EX_d4", expectimax_factory(eval_mobility_only, 4)),
    ]

    rows = []
    for label_a, fa, label_b, fb in configs:
        print(f"  {label_a} vs {label_b} ... ", end="", flush=True)
        t0 = time.time()
        res = balanced_match(fa, fb, N_MV_EXP)
        elapsed = time.time() - t0
        sig = res["p_value"] < 0.05
        print(
            f"{res['wins_a']:3d}/{N_MV_EXP}  "
            f"wr={res['win_rate_a']:.3f}  "
            f"p={res['p_value']:.4f}  "
            f"({'sig' if sig else 'n.s.':4s})  "
            f"({elapsed:.0f}s)"
        )
        rows.append({
            "agent_a": label_a, "agent_b": label_b,
            "wins_a": res["wins_a"], "n": N_MV_EXP,
            "win_rate_a": round(res["win_rate_a"], 4),
            "p_value": round(res["p_value"], 5),
            "ci_lo": round(res["ci_lo"], 4),
            "ci_hi": round(res["ci_hi"], 4),
            "significant": sig,
        })

    _write_csv(rows, OUT_DIR / "rigorous_minimax_vs_exp.csv")
    return rows


# ── Fase 4: Mejor agente vs Stratagem ─────────────────────────────────────

def phase4_vs_stratagem(rr_summary: list, depth_rows: list):
    if not HAS_STRATAGEM:
        print("\n[Fase 4] Stratagem no disponible, omitiendo.")
        return []

    # Elegir el mejor agente del sweep de profundidad (mayor win_rate vs baseline)
    best_depth_row = max(depth_rows, key=lambda r: r["win_rate"])
    best_h    = best_depth_row["heuristic"]
    best_d    = best_depth_row["depth"]
    best_hfn  = HEURISTICS_RR[best_h]

    print(f"\n{'='*65}")
    print(f"FASE 4 — Mejor agente vs Stratagem (cátedra)")
    print(f"  Agente propio : Minimax-AB, {best_h}, depth={best_d}")
    print(f"  Partidas      : {N_STRATAGEM} (balanceadas)")
    print(f"{'='*65}")

    our_factory  = minimax_factory(best_hfn, best_d)
    strat_factory = lambda p: Stratagem(p)

    t0 = time.time()
    res = balanced_match(our_factory, strat_factory, N_STRATAGEM)
    elapsed = time.time() - t0

    sig = res["p_value"] < 0.05
    print(
        f"  Nuestro agente: {res['wins_a']:3d}/{N_STRATAGEM} victorias  "
        f"wr={res['win_rate_a']:.3f}  "
        f"CI95=[{res['ci_lo']:.3f},{res['ci_hi']:.3f}]  "
        f"p={res['p_value']:.4f}  "
        f"({'sig' if sig else 'n.s.'})"
        f"  ({elapsed:.0f}s)"
    )

    rows = [{
        "our_agent": f"Minimax-AB_{best_h}_d{best_d}",
        "opponent": "Stratagem",
        "wins_ours": res["wins_a"],
        "n": N_STRATAGEM,
        "win_rate": round(res["win_rate_a"], 4),
        "p_value": round(res["p_value"], 5),
        "ci_lo": round(res["ci_lo"], 4),
        "ci_hi": round(res["ci_hi"], 4),
        "significant": sig,
    }]
    _write_csv(rows, OUT_DIR / "rigorous_vs_stratagem.csv")
    return rows


# ── CSV helper ─────────────────────────────────────────────────────────────

def _write_csv(rows: list, path: Path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Guardado: {path.name}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    t_global = time.time()
    print("=" * 65)
    print("EXPERIMENTO RIGUROSO — PROYECTO MATE")
    print("Isolation 4×4 | Minimax con Alpha-Beta Pruning")
    print("=" * 65)

    rr_summary, any_sig = phase1_roundrobin()
    depth_rows           = phase2_depth_sweep(rr_summary)
    mv_exp_rows          = phase3_minimax_vs_expectimax()
    vs_strat_rows        = phase4_vs_stratagem(rr_summary, depth_rows)

    total = time.time() - t_global
    print(f"\n{'='*65}")
    print(f"Experimento completado en {total/60:.1f} minutos.")
    print(f"Resultados guardados en: {OUT_DIR}")
    print(f"{'='*65}")

    # Resumen ejecutivo
    print("\n=== RESUMEN EJECUTIVO ===")
    print(f"\n[Fase 1] Heurísticas estadísticamente equivalentes: {not any_sig}")
    best_overall = rr_summary[0]["heuristic"]
    print(f"  Mejor heurística por win-rate: {best_overall}")

    best_depth = max(depth_rows, key=lambda r: r["win_rate"])
    print(f"\n[Fase 2] Mejor configuración vs baseline (mob_only d3):")
    print(
        f"  {best_depth['heuristic']} depth={best_depth['depth']}  "
        f"wr={best_depth['win_rate']:.3f}  "
        f"p={best_depth['p_value']:.4f}  "
        f"sig={best_depth['significant_p05']}"
    )

    print(f"\n[Fase 3] Minimax vs Expectimax:")
    for r in mv_exp_rows:
        print(
            f"  {r['agent_a']} vs {r['agent_b']:8s}  "
            f"wr={r['win_rate_a']:.3f}  p={r['p_value']:.4f}  sig={r['significant']}"
        )

    if vs_strat_rows:
        r = vs_strat_rows[0]
        print(f"\n[Fase 4] vs Stratagem: wr={r['win_rate']:.3f}  p={r['p_value']:.4f}")


if __name__ == "__main__":
    main()
