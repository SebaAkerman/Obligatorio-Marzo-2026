"""
Validacion escalada (~8h) — reproduce las corridas de alta potencia:
  - Round-robin heuristicas 1500/par (Bonferroni), depth sweep 500, MM vs EX ampliado
  - Agente final vs Stratagem, escalera de profundidad d5/d6
  - Dyna-Q multi-seed (q0/q1) y planning sweep multi-seed
Escribe CSVs incrementalmente en models/scaled_validation/ (no pisa los CSV principales
hasta revisar). Ejecutar desde Isolation/:  python ../scripts/experiment_scaled_validation.py
"""
import sys, os, time, csv, random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ISO = REPO / "Isolation"
MC = REPO / "MountainCarContinuous"
XL = REPO / "models" / "scaled_validation"
XL.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ISO))
sys.path.insert(0, str(MC))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import gymnasium as gym

import experiment_heuristics_rigorous as R
from expectimax_agent import ExpectimaxAgent
from minimax_agent import MinimaxAgent
from stratagem import Stratagem
from heuristics import eval_mobility_only, eval_territory, eval_mobility_territory
from dyna_q_agent import DynaQAgent
from q_learning_agent import QLearningAgent

t_start = time.time()
def log(msg):
    el = time.time() - t_start
    print(f"[{el/60:6.1f}min] {msg}", flush=True)

def wcsv(rows, name):
    if not rows: return
    with open(XL / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    log(f"  -> guardado {name}")

# Repuntar salida de R hacia XL y subir N
R.OUT_DIR = XL
R.N_RR = 1500
R.N_DEPTH = 500
R.N_STRATAGEM = 500

log("=== FASE 1: Round-robin 1500/par (depth 3) ===")
try:
    rr_summary, any_sig = R.phase1_roundrobin()
    log(f"Fase1 done. equivalentes={not any_sig}")
except Exception as e:
    log(f"Fase1 ERROR {e}"); rr_summary=[{"heuristic":"mob_only"}]; any_sig=False

log("=== FASE 2: Depth sweep 500/config ===")
try:
    depth_rows = R.phase2_depth_sweep(rr_summary)
except Exception as e:
    log(f"Fase2 ERROR {e}"); depth_rows=[{"heuristic":"mob_only","depth":4,"win_rate":0.6}]

log("=== FASE 4: vs Stratagem 500 (mob_only d4) ===")
try:
    R.phase4_vs_stratagem(rr_summary, depth_rows)
except Exception as e:
    log(f"Fase4 ERROR {e}")

# ---- FASE 3 custom: MM vs EX ampliado ----
log("=== FASE 3: MM vs EX (ampliado) ===")
try:
    mvx = []
    configs = [
        ("MM_d2", R.minimax_factory(eval_mobility_only,2), "EX_d2", R.expectimax_factory(eval_mobility_only,2), 300),
        ("MM_d3", R.minimax_factory(eval_mobility_only,3), "EX_d3", R.expectimax_factory(eval_mobility_only,3), 300),
        ("MM_d3", R.minimax_factory(eval_mobility_only,3), "EX_d4", R.expectimax_factory(eval_mobility_only,4), 200),
        ("MM_d4", R.minimax_factory(eval_mobility_only,4), "EX_d4", R.expectimax_factory(eval_mobility_only,4), 150),
    ]
    for la, fa, lb, fb, n in configs:
        res = R.balanced_match(fa, fb, n)
        row = {"agent_a":la,"agent_b":lb,"wins_a":res["wins_a"],"n":n,
               "win_rate_a":round(res["win_rate_a"],4),"p_value":round(res["p_value"],5),
               "ci_lo":round(res["ci_lo"],4),"ci_hi":round(res["ci_hi"],4),
               "significant":res["p_value"]<0.05}
        mvx.append(row); log(f"  {la} vs {lb}: {res['wins_a']}/{n} wr={res['win_rate_a']:.3f} p={res['p_value']:.4f}")
        wcsv(mvx, "rigorous_minimax_vs_exp.csv")
except Exception as e:
    log(f"Fase3 ERROR {e}")

# ---- FASE 4b: otros agentes vs Stratagem ----
log("=== FASE 4b: EX y territory vs Stratagem 300 ===")
try:
    strat = lambda p: Stratagem(p)
    extra = []
    for name, fac, n in [
        ("EX_mob_d3", R.expectimax_factory(eval_mobility_only,3), 300),
        ("MM_territory_d4", R.minimax_factory(eval_territory,4), 300),
        ("MM_mob_territory_d4", R.minimax_factory(eval_mobility_territory,4), 300),
    ]:
        res = R.balanced_match(fac, strat, n)
        extra.append({"our_agent":name,"opponent":"Stratagem","wins_ours":res["wins_a"],"n":n,
                      "win_rate":round(res["win_rate_a"],4),"p_value":round(res["p_value"],5),
                      "ci_lo":round(res["ci_lo"],4),"ci_hi":round(res["ci_hi"],4)})
        log(f"  {name} vs Stratagem: {res['wins_a']}/{n} wr={res['win_rate_a']:.3f}")
        wcsv(extra, "xl_vs_stratagem_others.csv")
except Exception as e:
    log(f"Fase4b ERROR {e}")

# ---- FASE 5: depth 5/6 ----
log("=== FASE 5: depth 5/6 exploracion ===")
try:
    strat = lambda p: Stratagem(p)
    d5 = []
    for label, fa, fb, n in [
        ("d5_vs_d4", R.minimax_factory(eval_mobility_only,5), R.minimax_factory(eval_mobility_only,4), 300),
        ("d5_vs_stratagem", R.minimax_factory(eval_mobility_only,5), strat, 300),
        ("d6_vs_d5", R.minimax_factory(eval_mobility_only,6), R.minimax_factory(eval_mobility_only,5), 60),
        ("d6_vs_stratagem", R.minimax_factory(eval_mobility_only,6), strat, 60),
    ]:
        res = R.balanced_match(fa, fb, n)
        d5.append({"matchup":label,"wins":res["wins_a"],"n":n,"wr":round(res["win_rate_a"],4),
                   "p":round(res["p_value"],5),"ci_lo":round(res["ci_lo"],4),"ci_hi":round(res["ci_hi"],4)})
        log(f"  {label}: {res['wins_a']}/{n} wr={res['win_rate_a']:.3f}")
        wcsv(d5, "xl_depth56.csv")
except Exception as e:
    log(f"Fase5 ERROR {e}")

# ---- FASE 6: Dyna-Q multi-seed (FINAL config 20x20x15) ----
log("=== FASE 6: Dyna-Q multi-seed ===")
try:
    def train_dyna(seed, q_init, episodes, n_plan=10):
        random.seed(seed); np.random.seed(seed)
        env = gym.make("MountainCarContinuous-v0"); env.reset(seed=seed); env.action_space.seed(seed)
        ag = DynaQAgent(20,20,15,n_planning_steps=n_plan,q_init=q_init)
        ag.train_agent(env, episodes=episodes, epsilon=1.0, gamma=0.99, alpha=0.2,
                       epsilon_decay=0.9995, epsilon_min=0.05, verbose=False)
        r = ag.test_agent(env, episodes=50); env.close()
        return r
    dyna = []
    plan = [("q0_5k",0.0,5000),("q1_5k",1.0,5000),("q0_10k",0.0,10000)]
    for tag,qi,eps in plan:
        for seed in range(5):
            r = train_dyna(seed, qi, eps)
            dyna.append({"config":tag,"q_init":qi,"episodes":eps,"seed":seed,
                         "mean_reward":round(r["mean_reward"],3),"std":round(r["std_reward"],3),
                         "success_rate":r["success_rate"]})
            log(f"  dyna {tag} seed{seed}: {r['mean_reward']:.1f} / {r['success_rate']:.0%}")
            wcsv(dyna, "xl_dyna_multiseed.csv")
except Exception as e:
    log(f"Fase6 ERROR {e}")

# ---- FASE 7: Dyna planning sweep multi-seed (el punto debil) ----
log("=== FASE 7: Dyna planning sweep multi-seed (q_init=1) ===")
try:
    def train_dyna_n(seed, n_plan, q_init=1.0, episodes=5000):
        random.seed(seed); np.random.seed(seed)
        env = gym.make("MountainCarContinuous-v0"); env.reset(seed=seed); env.action_space.seed(seed)
        ag = DynaQAgent(20,20,15,n_planning_steps=n_plan,q_init=q_init)
        ag.train_agent(env, episodes=episodes, epsilon=1.0, gamma=0.99, alpha=0.2,
                       epsilon_decay=0.9995, epsilon_min=0.05, verbose=False)
        r = ag.test_agent(env, episodes=50); env.close(); return r
    psweep = []
    for n_plan in [1,5,10,20,50]:
        for seed in range(3):
            r = train_dyna_n(seed, n_plan)
            psweep.append({"n_planning":n_plan,"seed":seed,"q_init":1.0,"episodes":5000,
                           "mean_reward":round(r["mean_reward"],3),"success_rate":r["success_rate"]})
            log(f"  planning n={n_plan} seed{seed}: {r['mean_reward']:.1f} / {r['success_rate']:.0%}")
            wcsv(psweep, "xl_dyna_planning_multiseed.csv")
except Exception as e:
    log(f"Fase7 ERROR {e}")

log(f"=== HEAVY RUN COMPLETO en {(time.time()-t_start)/60:.1f} min ===")
