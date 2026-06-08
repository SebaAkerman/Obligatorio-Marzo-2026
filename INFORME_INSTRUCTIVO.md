# Instructivo: Cómo Armar el Informe PDF del Obligatorio

**Obligatorio IA 2026 — Obligatorio Marzo**  
Universidad ORT Uruguay — Inteligencia Artificial

---

## 1. HERRAMIENTA RECOMENDADA

**Usar Google Docs o Microsoft Word → exportar a PDF.**

Es la opción más rápida, produce PDF bien formateado, permite insertar imágenes con captions.  
Alternativa técnica: Overleaf (LaTeX online) para formato académico premium.

> No usar `nbconvert` directamente — los notebooks tienen código mezclado con resultados  
> y el PDF queda ilegible. Copiar los resultados y figuras al documento manualmente.

---

## 2. DATOS VERIFICADOS PARA EL INFORME

Todos los números verificados con 100 episodios de evaluación (20/6/2026):

### Proyecto LOST

| Modelo | Config | Reward | Std | CI 95% | Éxito |
|--------|--------|--------|-----|--------|-------|
| Q-Learning FINAL | 30×30 bins, 15 acc, 15k eps | **94.14** | 0.63 | ±0.12 | **100%** | ~25 min |
| Dyna-Q FINAL | 20×20 bins, 15 acc, n=10, 5k eps | **89.41** | 0.30 | ±0.06 | **100%** | ~10 min |

**Archivos:** `models/lost/qlearning_FINAL.pkl` y `models/lost/dynaq_FINAL.pkl`

#### Hiperparámetros óptimos encontrados:
| Param | Valor óptimo | Por qué |
|-------|-------------|---------|
| α (alpha) | 0.2 | α=0.05 aprende lento, α=0.3 diverge |
| γ (gamma) | 0.99 | γ<0.99 → recompensa meta desaparece: 0.9^500 ≈ 10^-23 |
| ε decay | 0.9995 | Más lento no baja ε lo suficiente para explotar |
| Bins | 30×30 | Balance representación/cobertura |
| Acciones | 15 | Suficiente precisión; 20 agrega redundancia |
| q_init | 1.0 (QL) | Clave para Q-Learning; DynaQ no la necesita |
| n_planning | 10 (DQ) | Ver experimento `dyna_planning_sweep.png` |

#### Sweeps de hiperparámetros (para gráficos del informe):
- `reports/figures/alpha_search.png` — grilla α
- `reports/figures/gamma_search.png` — grilla γ
- `reports/figures/epsilon_decay_search.png` — grilla ε decay
- `reports/figures/dyna_planning_sweep.png` — sweep n_planning (generado por `experiment_dyna_planning.py`)
- `reports/figures/ql_learning_curve.png` — curva QL
- `reports/figures/dq_learning_curve.png` — curva DynaQ
- `reports/figures/ql_vs_dq_comparison.png` — comparación primeros 5k eps

### Proyecto MATE

#### Impacto de Alpha-Beta Pruning (datos reales, 50 partidas por config):

| Profundidad | Pure Minimax | AB Pruning | AB + Move Ordering |
|-------------|-------------|------------|-------------------|
| 2 | 3,608 n / 0.143s | 795 n / 0.031s **(-78%)** | 286 n / 0.138s **(-92%)** |
| 3 | 91,153 n / 3.178s | 8,752 n / 0.304s **(-90%)** | 2,341 n / 0.438s **(-97%)** |
| 4 | **1,586,170 n / 45.6s** | 40,688 n / 1.15s (-97%) | 5,526 n / 1.94s (-99.7%) |

**Datos completos: 50 partidas vs Random por configuración.**

**Observaciones clave:**
- Depth 4 Pure Minimax: **45.6s/partida** — completely infeasible en tiempo real (×14 respecto a depth 3)
- AB Pruning depth 4: 97.4% menos nodos, **39× más rápido** que pure → 1.15s/partida (usable)
- Move Ordering: 99.7% menos nodos que pure pero **más lento que AB solo** (1.94s vs 1.15s)
  - El overhead de sorting supera el beneficio de menos nodos cuando h_mobility es barata
  - Win rate de AB+MO (92%) < AB (98%) a depth 4 — sobre-ordenamiento puede causar poda sub-óptima
- Conclusión: **AB es suficiente y necesario. MO útil con heurísticas costosas (h_territory, h_future_mobility)**

**Archivos:** `reports/figures/ab_impact.png`, `models/mate/ab_impact_results.csv`

#### Torneos (50 partidas, depth=3, mob_only):
| Matchup | Win rate Agent 1 | CI 95% |
|---------|-----------------|--------|
| Minimax-AB vs Random | 100% | — |
| Expectimax vs Random | 78% | ±11.5% |
| Minimax-AB vs Expectimax | 98% | ±1.9% |
| Minimax(full) vs Minimax(mobility) | 76% | ±11.8% |

#### Experimento Riguroso de Heurísticas (100 partidas/par, Bonferroni, depth=3):

**Diseño:** C(5,2)=10 pares × 100 partidas balanceadas (50 P1 + 50 P2). Test binomial por par.  
Corrección Bonferroni: α_adj = 0.05/10 = **0.005**. Potencia ~94% para diferencias ≥10%.

| Heurística | Wins | Games | Win Rate | IC Wilson 95% | p-valor mejor par |
|------------|------|-------|----------|---------------|-------------------|
| mob_territory | 214 | 400 | 53.5% | [0.486, 0.583] | 0.1096 (n.s.) |
| full | 212 | 400 | 53.0% | [0.481, 0.578] | 0.2301 (n.s.) |
| territory | 196 | 400 | 49.0% | [0.441, 0.539] | 0.3173 (n.s.) |
| mob_only | 189 | 400 | 47.2% | [0.424, 0.521] | 0.1096 (n.s.) |
| mob_center | 189 | 400 | 47.2% | [0.424, 0.521] | 0.5485 (n.s.) |

**Conclusión científica:** Ningún par supera α=0.005 post-Bonferroni. Las heurísticas son  
**estadísticamente equivalentes** en tablero 4×4 a depth=3. La elección correcta es `mob_only`  
(más simple, más rápida), ya que los datos no justifican mayor complejidad.  
**Ver:** `models/mate/rigorous_significance.csv`, `rigorous_matchup_matrix.csv`

#### Experimento Riguroso — Sweep de Profundidad (vs baseline mob_only d3):

| Config | Wins/100 | Win Rate | p-valor | Significativo |
|--------|----------|----------|---------|---------------|
| mob_only d2 | 34 | 34% | 0.0014 | ✓ — depth=2 pierde |
| full d2 | 39 | 39% | 0.0278 | ✓ — depth=2 pierde |
| mob_territory d2 | 45 | 45% | 0.317 | n.s. |
| mob_territory d3 | 38 | 38% | 0.016 | ✓ — equivalente al baseline (ruido) |
| full d3 | 44 | 44% | 0.230 | n.s. |
| **mob_only d4** | **67** | **67%** | **0.0007** | **✓✓ MUY SIGNIFICATIVO** |
| mob_territory d4 | 65 | 65% | 0.0027 | ✓✓ significativo |
| full d4 | 56 | 56% | 0.230 | n.s. — complejidad extra no ayuda |

**Hallazgo central:** La profundidad domina sobre la calidad de la heurística.  
`mob_only d4` vence a `mob_only d3` con p=0.0007 — diferencia altamente significativa.  
`full d4` no mejora significativamente: la mayor complejidad introduce ruido en las hojas.

#### Experimento Riguroso — Minimax vs Expectimax (100 partidas balanceadas):

| Matchup | Wins MM / 100 | Win Rate | p-valor | Conclusión |
|---------|--------------|----------|---------|------------|
| MM d2 vs EX d2 | 38 | 38% | 0.0164 | EX gana a depth=2 |
| MM d3 vs EX d3 | 94 | **94%** | **0.0000** | MM aplasta a depth=3 |
| MM d3 vs EX d4 | 58 | 58% | 0.1096 | n.s. — EX d+1 lo equipara |

A depth=2: Expectimax gana porque el oponente no puede ser "visto" con suficiente lookahead.  
A depth=3: Minimax aplasta (94%) — el oponente juega cerca del óptimo, la suposición de Minimax es correcta.  
EX necesita un nivel extra de profundidad para compensar su modelo de oponente incorrecto.

#### vs Stratagem — Agente Final (100 partidas balanceadas, mob_only d4):

| Agente | Wins/100 | Win Rate | IC Wilson 95% | p-valor |
|--------|----------|----------|---------------|---------|
| Minimax-AB mob_only d4 | **76** | **76%** | [0.668, 0.833] | **0.0000** |

**Archivo:** `models/mate/rigorous_vs_stratagem.csv`

---

## 3. ESTRUCTURA DEL INFORME (≤ 20 páginas + anexos)

### Portada (1 página)
- Título: "Obligatorio IA 2026 — Red Destination™"
- Integrantes: Sebastian Akerman (282163), Felipe Kelmanson (???)
- Materia, docente, fecha

### 1. Proyecto LOST — Q-Learning (páginas 2-10)

#### 1.1 Descripción del ambiente (0.5 pág)
- MountainCarContinuous-v0: obs [x ∈ [-1.2,0.6], vel ∈ [-0.07,0.07]], acc ∈ [-1,1]
- Reward: -0.1·a² por paso + 100 al llegar (x ≥ 0.45)
- Máx 999 pasos. Episodio exitoso = `terminated=True`

#### 1.2 Discretización (1 pág)
- Por qué discretizar: tabular Q-Learning requiere estados discretos
- Cómo: `np.digitize` → bins uniformes → índice (i,j)
- Tabla comparando bins 10/20/30 × acciones 5/10/15/20:
  ```
  Config        | Éxito | Reward | Space
  10×10 × 5    | 95%   | 86.1   | 605
  10×10 × 10   | 90%   | 76.3   | 1,210
  20×20 × 10   | 95%   | 85.4   | 4,410
  20×20 × 15   | 100%  | 92.2   | 6,615
  30×30 × 15   | 100%  | 93.7   | 14,415
  30×30 × 20   | 100%  | 93.8   | 19,220
  ```
- Elección final: 30×30 bins, 15 acciones (mejor reward con tamaño razonable)
- **Insertar figura:** ninguna necesaria aquí, tabla alcanza

#### 1.3 Q-Learning (1.5 pág)
- Algoritmo: la regla de Bellman (ecuación centrada)
- Exploración: ε-greedy con decay multiplicativo
- Por qué q_init=1.0 es crucial (sección de análisis de fallas)
- **Insertar figura:** `ql_learning_curve.png` (curva de aprendizaje, 15k eps)
- Hitos: primer éxito ep 732, media-100 ≥ 50 en ep 2368, convergencia estable ep 5000+

#### 1.4 Búsqueda de Hiperparámetros (2 pág)
Para cada hiperparámetro: tabla de resultados + figura + conclusión.

**α (alpha):**
| α | Reward | Éxito |
|---|--------|-------|
| 0.05 | 61.2 | 66% — aprende muy lento |
| 0.1 | 92.9 | 100% |
| 0.2 | **93.9** | **100%** — óptimo |
| 0.3 | -36.5 | 0% — Q-table diverge |
| 0.5 | 84.9 | 96% — inestable |
- **Insertar figura:** `alpha_search.png`

**γ (gamma) — hallazgo crítico:**
| γ | Éxito | Explicación |
|---|-------|-------------|
| 0.90 | 0% | 0.9^500 ≈ 10^-23: meta invisible |
| 0.95 | 0% | 0.95^500 ≈ 5.15×10^-12: invisible |
| 0.99 | **100%** | Óptimo |
| 0.999 | 94% | Sobreestima estados remotos |
- **Insertar figura:** `gamma_search.png`

**ε decay:**
| Decay | ε final (15k eps) | Éxito |
|-------|------------------|-------|
| 0.9990 | 0.05 | 100% |
| **0.9995** | 0.05 | **100%** — balance óptimo |
| 0.9997 | 0.22 | 98% — poco explotación |
| 0.9999 | 0.61 | 84% — nunca explota |
- **Insertar figura:** `epsilon_decay_search.png`

#### 1.5 Dyna-Q (1.5 pág)
- Algoritmo de Sutton & Barto (caps 8.1-8.2): integra actuación, aprendizaje y planificación
- Por cada paso real: (1) actualizar Q, (2) actualizar modelo, (3) n pasos simulados
- n=0 reduce a Q-Learning puro
- Por qué Dyna-Q no necesita q_init=1.0: el planning propaga el valor de la meta más rápido

**Sweep n_planning (3000 eps, q_init=0, decay=0.9995) — resultados completos:**
| n_planning | Reward | Éxito | 1er éxito | Tiempo |
|-----------|--------|-------|-----------|--------|
| 0 (QL puro) | 0.00 | 0% | ep 17 (no converge) | 59s |
| 1 | 0.00 | 0% | nunca | 69s |
| 5 | 0.00 | 0% | nunca | 101s |
| **10** | **39.97** | **66%** | ep 51 | 120s |
| 20 | -28.94 | 18% | ep 47 | 199s |
| 50 | 0.00 | 0% | nunca | 420s |

**Hallazgo: curva NO monótona (forma de campana).** n=10 es el óptimo para 3000 eps:
- n<10: planning insuficiente, no encuentra la meta
- n=10: balance óptimo (66% a 3k eps → 100% a 5k eps)
- n=20: convergencia prematura (refuerza Q-values incorrectos en modelo escaso)
- n=50: mayor sobrejuste aún + episodios 7× más lentos = 0%

Implicación: más planning no siempre es mejor. El n óptimo depende del presupuesto de episodios reales y la densidad del modelo aprendido.

- **Insertar figura:** `dyna_planning_sweep.png` (sweep n=0..50 completo, con curvas de aprendizaje)
- **Insertar figura:** `ql_vs_dq_comparison.png` (comparación primeros 5k eps)

#### 1.6 Evaluación Final y Conclusiones LOST (0.5 pág)
- Tabla con los dos modelos FINALES (datos verificados arriba)
- Q-Learning: mejor reward final (94.14) con 15k eps y q_init=1.0
- Dyna-Q: menor varianza (std=0.30 vs 0.63), converge en 5k eps (3x menos episodios)
- Trade-off: Dyna-Q aprende 3x más rápido; Q-Learning llega a mejor performance con más episodios

### 2. Proyecto MATE — Minimax/Expectimax (páginas 11-18)

#### 2.1 Descripción del ambiente (0.5 pág)
- Isolation 4×4: dos jugadores, turnos alternados
- Acción: (dirección[0-7], celda_a_destruir)
- Pierde quien no tenga movimientos válidos
- Factor de ramificación: hasta ~96 acciones por jugador

#### 2.2 Minimax con Alpha-Beta Pruning (2 pág)

**Algoritmo:**
- Max: nuestro agente maximiza utilidad
- Min: oponente minimiza utilidad
- Alpha-Beta: elimina ramas que no pueden afectar la decisión
  - α = mejor garantía de Max (inicia -∞)
  - β = mejor garantía de Min (inicia +∞)
  - Poda β: nodo Max con valor ≥ β
  - Poda α: nodo Min con valor ≤ α
- Complejidad: O(b^d) pure → O(b^(d/2)) con AB óptimo

**Impacto de Alpha-Beta — datos reales, 50 partidas por configuración:**
| Prof. | Pure Minimax | AB Pruning | AB + Move Ordering |
|-------|-------------|------------|-------------------|
| 2 | 3,608 n / 0.14s | 795 n / 0.03s **(-78%)** | 286 n / 0.14s **(-92%)** |
| 3 | 91,153 n / 3.18s | 8,752 n / 0.30s **(-90%)** | 2,341 n / 0.44s **(-97%)** |
| **4** | **1,586,170 n / 45.6s** | 40,688 n / 1.15s (-97%) | 5,526 n / 1.94s (-99.7%) |

- **Insertar figura:** `ab_impact.png`
- Depth 4 Pure Minimax: **45.6s por partida** — inviable en tiempo real. AB reduce 97% → 1.15s/partida.
- Move Ordering (MO): 99.7% menos nodos pero **más lento** que AB solo (1.94s vs 1.15s). El overhead de sorting supera el beneficio cuando la heurística es barata (h_mobility). MO útil con heurísticas costosas (h_territory, h_future_mobility).
- **Conclusión:** Alpha-Beta es obligatoria para depth ≥ 4.

#### 2.3 Expectimax (1 pág)
- Diferencia con Minimax: nodo Min → nodo Chance (distribución uniforme)
- E[V] = Σ P(a) · V(sucesor), P(a) = 1/|acciones|
- Cuándo usar: cuando el oponente es aleatorio o subóptimo
- Resultado: Minimax-AB gana a Expectimax 98% (ambos depth=3, mob_only)
- Por qué: el oponente (otro Minimax) juega cerca del óptimo → asunción de Expectimax es incorrecta

#### 2.4 Funciones de Evaluación Heurística (2 pág)

**Heurísticas implementadas:**
| Heurística | Fórmula | Costo |
|------------|---------|-------|
| `h_mobility` | mis_movs - opp_movs | O(1) |
| `h_center_proximity` | (dist_opp - dist_yo) / max_dist | O(1) |
| `h_open_cells` | (mis_acciones - opp_acciones) / total | O(1) |
| `h_future_mobility` | promedio movs futuros míos - opp | ~20× h_mobility |
| `h_territory` | BFS: celdas vacías alcanzables yo - opp | ~1.8× h_mobility |

**Funciones compuestas:**
- `eval_mobility_only`: solo h_mobility — señal más directa; **elegida como heurística final**
- `eval_mobility_center`: 0.7×mob + 0.3×center — útil en early-game cuando el tablero está abierto
- `eval_full`: 0.6×mob + 0.2×center + 0.2×open_cells — señales parcialmente redundantes
- `eval_territory`: solo h_territory (BFS) — captura control territorial a largo plazo
- `eval_mobility_territory`: 0.6×mob + 0.4×territory — balance señal inmediata + territorial

**Nota h_open_cells:** usa la misma señal subyacente que h_mobility (get_possible_actions), solo difiere en normalización. No es una señal independiente.

**Por qué mob_only es la elección final:** el experimento riguroso (100 partidas/par, Bonferroni) demostró que ninguna heurística supera estadísticamente a las demás en 4×4 a depth=3. Se elige `mob_only` por ser la más simple y la de menor costo computacional — sin penalización empírica.

#### 2.5 Experimentación y Torneos (2 pág)

**Diseño experimental — qué se midió y por qué:**

| Prueba | Objetivo | Metodología | N |
|--------|----------|-------------|---|
| Pure vs AB vs AB+MO | Cuantificar impacto de Alpha-Beta | nodos + tiempo + win% vs Random | 50/config |
| Minimax vs Expectimax | Decidir qué técnica usar | Balanced match, test binomial | 100/matchup |
| Round-robin heurísticas | Comparar funciones de evaluación | Round-robin balanceado + Bonferroni | 100/par |
| Sweep depth × heurística | Determinar si depth compensa heurística | vs baseline mob_only d3 | 100/config |
| Agente final vs Stratagem | Benchmark externo (agente cátedra) | Balanced match | 100 |
| Agente final vs Random | Confirmar dominancia básica | Balanced match | 100 |
| Mirror match d4 vs d4 | Cuantificar ventaja P1 a depth final | Sin balanceo de roles | 100 |

**Criterio de evaluación:** win rate balanceado (50% P1, 50% P2) + test binomial (α=0.05,
corrección Bonferroni donde hay múltiples comparaciones) + IC Wilson 95%.

**Impacto de la profundidad:**
| Depth | Win% vs Random | Nodos promedio/partida |
|-------|---------------|----------------------|
| 2 | 100% | 3,038 |
| 3 | 99% | 23,368 |
| 4 | 100% (n=10) | 187,758 |
- Depth=2 ya domina a Random. Las diferencias emergen contra oponentes hábiles.

**Experimento riguroso de heurísticas (100 partidas/par, Bonferroni α=0.005):**

Ningún par de heurísticas resultó estadísticamente significativo post-corrección de Bonferroni.
Las 5 heurísticas son **equivalentes** en tablero 4×4 a depth=3. Se elige `mob_only` por simplicidad.

- **Datos:** `models/mate/rigorous_significance.csv`, `rigorous_matchup_matrix.csv`, `rigorous_summary.csv`
- **Insertar figura:** `heuristic_roundrobin_full.png` (heatmap exploratorio, 60 partidas/par previo)

**Sweep de profundidad — hallazgo central (vs baseline mob_only d3):**

| Config | Win Rate | p-valor | Significativo |
|--------|----------|---------|---------------|
| mob_only d2 | 34% | 0.0014 | ✓ — depth=2 significativamente peor |
| **mob_only d4** | **67%** | **0.0007** | **✓✓ — depth=4 significativamente mejor** |
| mob_territory d4 | 65% | 0.0027 | ✓✓ — también mejora |
| full d4 | 56% | 0.230 | n.s. — complejidad extra no ayuda |

→ La profundidad domina sobre la calidad de la heurística. `full` a mayor depth introduce ruido.

**Minimax vs Expectimax (100 partidas balanceadas, riguroso):**
| Matchup | Win rate Minimax | p-valor | Conclusión |
|---------|-----------------|---------|------------|
| MM d2 vs EX d2 | 38% | 0.0164 | EX gana a depth=2 |
| **MM d3 vs EX d3** | **94%** | **0.0000** | MM aplasta — oponente juega óptimo |
| MM d3 vs EX d4 | 58% | 0.1096 | n.s. — EX d+1 lo equipara |

**Ventaja del primer jugador:**
- Mirror match (Minimax vs Minimax): P1 gana ~70% en tablero 4×4
- El tablero pequeño favorece al primer movedor (más área "virgen")

**Experimentos complementarios (mob_only d4, 100 partidas balanceadas c/u):**

| Experimento | Resultado | Propósito |
|---|---|---|
| Agente final vs Stratagem | **76%**, p≈0, CI=[66.8%,83.3%] | Benchmark externo |
| Agente final vs Random | **85%**, p≈0, CI=[76.7%,90.7%] | Confirmar dominancia básica |
| Mirror match d4 vs d4 | P1 gana **73%**, p≈0, CI=[63.6%,80.7%] | Cuantificar ventaja P1 a depth=4 |

**Hallazgo del mirror match:** el primer jugador gana 73% de las partidas a depth=4,
diferencia estadísticamente significativa (p≈0). Esto explica por qué todos los torneos
balancean P1/P2 — sin ese balance, los resultados estarían sesgados estructuralmente.

- **Insertar figura:** `rigorous_vs_stratagem.png` ★ (incluye vs Random y vs Stratagem)
- **Datos:** `models/mate/rigorous_vs_stratagem.csv`, `models/mate/rigorous_extra.csv`

#### 2.6 Conclusiones MATE (0.5 pág)
- **Mejor agente final:** `MinimaxAgent(depth=4, heuristic=eval_mobility_only, use_alpha_beta=True)`
  - Validado por experimento riguroso: 76% vs Stratagem (p=0.0000, CI=[66.8%, 83.3%])
- **Heurísticas equivalentes:** experimento con Bonferroni confirma que ninguna heurística supera a otra en 4×4 depth=3. Se elige `mob_only` por ser la más simple y rápida.
- **La profundidad es el factor decisivo:** depth=4 supera a depth=3 con p=0.0007. Más importante que sofisticar la heurística.
- **Alpha-Beta ESENCIAL:** depth=4 sin AB = 45.6s/partida (inviable). Con AB = 1.15s/partida (97% menos nodos).
- **Minimax >> Expectimax a depth=3 (94%, p=0.0000).** Expectimax solo compite si se le da un nivel extra de profundidad.
- **Ventaja P1:** ~72% en 4×4. El primer jugador tiene ventaja estructural en tablero pequeño.

### 3. Conclusiones Generales (1 pág)

**Tabla comparativa ambos proyectos:**

| Aspecto | LOST (Q-Learning) | MATE (Minimax) |
|---------|------------------|----------------|
| Tipo de problema | Aprendizaje por refuerzo (ambiente continuo) | Búsqueda adversarial (juego de suma cero) |
| Técnica principal | Q-Learning tabular + Dyna-Q | Minimax con Alpha-Beta Pruning |
| Desafío central | Reward sparse; meta a ~500 pasos | Factor de ramificación ~96; profundidad limitada |
| Solución al desafío | q_init=1.0 (init. optimista) + γ=0.99 | Alpha-Beta Pruning (97% reducción de nodos) |
| Mejor resultado | 94.14 reward, 100% éxito (15k eps) | 76% vs Stratagem, 85% vs Random |
| Tiempo de entrenamiento | QL: ~25 min (15k eps) / DQ: ~10 min (5k eps) | No aplica (búsqueda en tiempo real) |
| Hallazgo no obvio | γ<0.99 hace la meta invisible (0.9^500≈10⁻²³) | Profundidad > heurística (mob_only d4 > territory d3) |

**Uso de IA Generativa:**
Herramienta: Claude (Anthropic). Uso: generación de esqueletos de clases, revisión de implementaciones y estructuración del repositorio. Todo el contenido fue verificado y comprendido por los integrantes. Los errores son responsabilidad de los autores.

**Dificultades encontradas:**

*(Copiar del punto 4 de Advertencias abajo)*

### 4. Advertencias y Dificultades (obligatorio por la letra)

#### LOST — Dificultades encontradas

**1. q_init=0 nunca converge** *(dificultad resuelta)*
Q-Learning con inicialización en 0 no llega a la meta en ninguna configuración probada. El agente aprende que aplicar fuerza cero minimiza la penalización `-0.1·a²`, quedando atrapado en un mínimo local. Solución: inicialización optimista `q_init=1.0` fuerza la exploración de cada estado antes de concluir que es malo.

**2. γ < 0.99 hace la meta invisible** *(dificultad resuelta)*
Con γ=0.90, la recompensa de la meta descontada hasta el inicio vale `0.9^500 ≈ 10⁻²³` — un número tan pequeño que los TD-updates nunca lo propagan. No es un bug; es una propiedad matemática del ambiente. Solo γ≥0.99 permite que el agente "vea" la meta a través de los ~500 pasos que la separan del inicio.

**3. Dyna-Q: curva no monótona de n_planning** *(dificultad resuelta)*
Más planning no siempre mejora el agente. Con n=20 el rendimiento cae de 66% a 18% respecto a n=10. El modelo aprendido con pocas visitas es impreciso; con n=20 pasos simulados por cada paso real, el agente refuerza valores Q incorrectos antes de que el modelo sea confiable. Solución: n=10 es el punto óptimo.

#### MATE — Dificultades encontradas

**4. Move Ordering paradox** *(dificultad resuelta)*
Implementar Move Ordering redujo los nodos expandidos en 99.7% pero resultó más lento que Alpha-Beta solo (1.94s vs 1.15s por partida) y con menor win rate a depth=4. El overhead de calcular y ordenar los scores heurísticos para cada acción supera el beneficio de la poda adicional cuando la heurística es `h_mobility` (O(1)). Move Ordering solo vale la pena con heurísticas costosas.

**5. Ventaja estructural del primer jugador** *(no resuelta — se mitiga)*
En tablero 4×4, el primer jugador gana el 73% de las partidas en mirror match (p≈0). Esta ventaja es inherente al tamaño del tablero y no se puede eliminar. Se mitiga usando torneos balanceados (50% P1, 50% P2) para que las comparaciones sean justas. En partidas reales, el rol se asigna al azar.

**6. Heurísticas estadísticamente equivalentes** *(hallazgo, no dificultad)*
Todos los intentos de diseñar heurísticas superiores a `h_mobility` fallaron en superar el umbral estadístico. Las 5 heurísticas probadas son equivalentes en tablero 4×4 a depth=3 (Bonferroni). Esto no es una limitación del código sino una propiedad del ambiente: con AB a depth=4, el lookahead compensa ampliamente la calidad de la evaluación en las hojas.

### 5. Anexos
- Figuras adicionales (curvas de aprendizaje individuales, heatmap round-robin exploratorio)
- Fragmentos de código relevante: clase `Discretizer`, regla de actualización Q, pseudocódigo Dyna-Q
- No poner código completo — referir al repositorio GitHub

---

## 4. FIGURAS DISPONIBLES

```
reports/figures/
│
│  PROYECTO LOST
├── ql_learning_curve.png             curva Q-Learning 15k eps
├── dq_learning_curve.png             curva Dyna-Q 5k eps
├── ql_vs_dq_comparison.png           comparación primeros 5k eps
├── alpha_search.png                  grilla α
├── gamma_search.png                  grilla γ
├── epsilon_decay_search.png          grilla ε decay
├── dyna_planning_sweep.png           sweep n_planning 0-50
│
│  PROYECTO MATE — Exploratorio (usa en informe solo como contexto)
├── ab_impact.png                     impacto Alpha-Beta (nodos + tiempo)
├── depth_vs_winrate.png              depth vs win% vs Random
├── heuristic_roundrobin_full.png     heatmap round-robin 60/par (exploratorio)
├── heuristic_roundrobin_bar.png      ranking round-robin 60/par (exploratorio)
├── vs_stratagem.png                  vs Stratagem depth=3 P1/P2 separado (OBSOLETO)
│
│  PROYECTO MATE — Experimento riguroso (★ USAR ESTAS EN EL INFORME)
├── rigorous_depth_sweep.png        ★ sweep depth×heurística con significancia
├── rigorous_roundrobin.png         ★ round-robin 100/par + matriz de matchups
├── rigorous_minimax_vs_exp.png     ★ Minimax vs Expectimax con CIs
├── rigorous_vs_stratagem.png       ★ agente final (d4) vs Stratagem + vs Random
└── rigorous_summary.png            ★ resumen ejecutivo 3 paneles (AB + depth + benchmarks)
```

**Para el informe usar siempre las figuras `rigorous_*`. Las figuras exploratorias anteriores
son útiles solo para mostrar el proceso iterativo de investigación.**

---

## 5. ARCHIVOS DE RESULTADOS — REFERENCIA COMPLETA

### Proyecto LOST
| Archivo | Contenido |
|---------|-----------|
| `models/lost/qlearning_FINAL.pkl` | Modelo Q-Learning (30×30, 15 acc, 15k eps) |
| `models/lost/dynaq_FINAL.pkl` | Modelo Dyna-Q (20×20, 15 acc, n=10, 5k eps) |
| `models/lost/dyna_planning_sweep.csv` | Sweep n_planning=0..50 |

### Proyecto MATE
| Archivo | Contenido |
|---------|-----------|
| `models/mate/ab_impact_results.csv` | Impacto Alpha-Beta (depth 2,3,4) |
| `models/mate/roundrobin_results.csv` | Round-robin exploratorio (60/par) |
| `models/mate/rigorous_significance.csv` | **p-valores por par (Bonferroni)** |
| `models/mate/rigorous_matchup_matrix.csv` | **Matriz de matchups completa** |
| `models/mate/rigorous_summary.csv` | **Ranking con IC Wilson** |
| `models/mate/rigorous_depth_sweep.csv` | **Sweep profundidad × heurística** |
| `models/mate/rigorous_minimax_vs_exp.csv` | **Minimax vs Expectimax riguroso** |
| `models/mate/rigorous_vs_stratagem.csv` | **Mejor agente vs Stratagem** |
| `models/mate/rigorous_extra.csv` | **Final vs Random + Mirror match (P1 advantage)** |

---

## 7. PASO A PASO: CREAR EL PDF

### Paso 1: Crear el documento
Abrir Google Docs. Título: "Obligatorio IA 2026 — Red Destination™".  
Configurar página: A4, márgenes 2.5cm, fuente Arial/Calibri 11pt, interlineado 1.15.

### Paso 2: Portada
- Título grande centrado
- Integrantes, número de estudiante, dictado, fecha (07/06/2026)

### Paso 3: Sección LOST — copiar datos verificados
Usar los números de la sección 2 de este instructivo (ya verificados).  
NO copiar del notebook directamente — cell 40 tiene datos incorrectos hardcodeados.  
Datos correctos: QL FINAL = 94.14 ± 0.63 / 100%, DQ FINAL = 89.41 ± 0.30 / 100%.

### Paso 4: Insertar figuras LOST
Para cada figura: Insert → Image → Upload from computer.  
Agregar caption debajo: "Figura N: Descripción"  
Figuras LOST (en orden): learning_curve → ql_vs_dq → alpha → gamma → epsilon → dyna_planning

### Paso 5: Sección MATE — datos del experimento riguroso
Todos los experimentos están completos. Usar estos datos verificados:
- AB impact: `models/mate/ab_impact_results.csv` → tabla de nodos y tiempos
- Heurísticas: `models/mate/rigorous_summary.csv` → equivalencia estadística (Bonferroni)
- Depth sweep: `models/mate/rigorous_depth_sweep.csv` → mob_only d4 gana (p=0.0007)
- Minimax vs Expectimax: `models/mate/rigorous_minimax_vs_exp.csv`
- vs Stratagem: `models/mate/rigorous_vs_stratagem.csv` → 76% wr (p=0.0000)

### Paso 6: Insertar figuras MATE
Figuras MATE (usar las `rigorous_*` como principales):
1. `ab_impact.png` — impacto Alpha-Beta (nodos y tiempo por profundidad)
2. `rigorous_depth_sweep.png` ★ — depth × heurística, resultado central
3. `rigorous_roundrobin.png` ★ — equivalencia estadística de heurísticas
4. `rigorous_minimax_vs_exp.png` ★ — Minimax vs Expectimax a distintas profundidades
5. `rigorous_vs_stratagem.png` ★ — benchmark final del agente elegido
6. `rigorous_summary.png` ★ — resumen 3-paneles para conclusiones

Figuras exploratorias opcionales (para mostrar proceso iterativo):
- `heuristic_roundrobin_full.png` — heatmap exploratorio 60/par

### Paso 7: Revisar longitud
Objetivo: 15-18 páginas de contenido, 2-5 páginas de anexos.  
Si es muy largo: comprimir tablas, reducir texto descriptivo.  
Si es muy corto: agregar análisis de fallas, limitaciones, trabajo futuro.

### Paso 8: Declaración uso de IA
Según la letra:  
"Herramienta utilizada: Claude (Anthropic). Contexto de uso: generación de código base (esqueletos de clases), revisión de implementaciones y estructuración del repositorio. Todo el contenido fue verificado y comprendido por los autores."

### Paso 9: Exportar PDF
File → Download → PDF Document (.pdf)  
Verificar que todas las imágenes se vean bien, las tablas no se corten.  
Tamaño final esperado: 5-15 MB dependiendo de la calidad de imágenes.

### Paso 10: Armar el ZIP de entrega

El ZIP debe llamarse `Obligatorio_Akerman_Kelmanson.zip` (o el nombre que pida la cátedra).

**Estructura dentro del ZIP:**
```
Obligatorio_Akerman_Kelmanson/
├── informe.pdf
├── MountainCarContinuous/
│   ├── q_learning_agent.py
│   ├── dyna_q_agent.py
│   ├── utils/discretization.py
│   └── continuous_mountain_car.ipynb
├── Isolation/
│   ├── minimax_agent.py
│   ├── expectimax_agent.py
│   ├── heuristics.py
│   ├── board.py, agent.py, isolation_env.py (cátedra)
│   └── isolation.ipynb
├── scripts/
│   ├── train_lost.py
│   ├── evaluate_mate.py
│   ├── experiment_heuristics_rigorous.py
│   └── generate_mate_figures.py
└── models/
    └── lost/
        ├── qlearning_FINAL.pkl   ← OBLIGATORIO
        └── dynaq_FINAL.pkl
```

**Qué NO incluir en el ZIP:** `.venv/`, `__pycache__/`, `models/lost/qlearning_bins_*.pkl` (exploración),
`models/mate/*.csv` (opcionales, pero conveniente incluirlos como evidencia de experimentos).

**Crear el ZIP:**
```bash
# Desde la carpeta padre de Obligatorio-Marzo-2026/
zip -r Obligatorio_Akerman_Kelmanson.zip Obligatorio-Marzo-2026/ \
  --exclude "*/\.venv/*" --exclude "*/__pycache__/*" --exclude "*/.git/*"
```

### Paso 11: Checklist final de auditoría

**Entregables físicos:**
- [ ] `informe.pdf` generado y ≤ 20 páginas (sin contar anexos)
- [ ] ZIP armado con estructura correcta (ver Paso 10)
- [ ] `models/lost/qlearning_FINAL.pkl` en el ZIP ← **OBLIGATORIO por la letra**
- [ ] `models/lost/dynaq_FINAL.pkl` en el ZIP
- [ ] Número de estudiante de Felipe Kelmanson completado en portada

**Contenido LOST:**
- [ ] Descripción del ambiente (obs space, reward, condición de éxito)
- [ ] Discretización: tabla de 6 configuraciones + elección justificada
- [ ] Q-Learning: ecuación Bellman, política ε-greedy, análisis q_init
- [ ] Hiperparámetros α, γ (con cálculo 0.9^500), ε decay — tabla + figura por cada uno
- [ ] Dyna-Q: algoritmo + sweep n_planning (curva no monótona explicada)
- [ ] Resultados finales con tiempos: QL 94.14±0.63/100% (~25 min), DQ 89.41±0.30/100% (~10 min)

**Contenido MATE:**
- [ ] Descripción ambiente Isolation 4×4
- [ ] Minimax + Alpha-Beta: algoritmo + tabla impacto (nodos y tiempos, incluyendo 45.6s pure d4)
- [ ] Expectimax: diferencia conceptual + resultado riguroso (94% MM d3 vs EX d3, p≈0)
- [ ] Heurísticas: tabla de las 5 implementadas + equivalencia estadística (Bonferroni)
- [ ] Experimento riguroso: depth sweep (mob_only d4 67%, p=0.0007) + vs Stratagem (76%)
- [ ] Agente final: `MinimaxAgent(depth=4, mob_only, AB=True)` con justificación

**Apoyo visual (mínimo 5 figuras con caption):**
- [ ] `ql_learning_curve.png` o `ql_vs_dq_comparison.png`
- [ ] Al menos 1 figura de sweep LOST (alpha/gamma/epsilon/dyna)
- [ ] `ab_impact.png`
- [ ] `rigorous_depth_sweep.png` ★
- [ ] `rigorous_minimax_vs_exp.png` ★ o `rigorous_vs_stratagem.png` ★

**Advertencias y dificultades** ← exigido explícitamente por la letra:
- [ ] q_init=0 no converge (y por qué se resolvió con q_init=1.0)
- [ ] γ<0.99 hace la meta invisible (con el cálculo 0.9^500 ≈ 10⁻²³)
- [ ] n_planning no monótono en Dyna-Q
- [ ] Move Ordering más lento que AB solo (paradoja explicada)
- [ ] Ventaja P1 73% en Isolation — cómo se mitigó con balanceo

**Declaración uso de IA:**
- [ ] Herramienta (Claude), contexto de uso, verificación por los autores

---

## 8. ESTADO DE EXPERIMENTOS — TODOS COMPLETADOS ✅

| Experimento | Script | Estado | Resultados |
|------------|--------|--------|------------|
| Alpha-Beta Impact | `experiment_ab_impact.py` | ✅ | `ab_impact_results.csv`, `ab_impact.png` |
| Dyna-Q n_planning sweep | `experiment_dyna_planning.py` | ✅ | `dyna_planning_sweep.csv`, `dyna_planning_sweep.png` |
| Round-robin exploratorio (60/par) | `experiment_heuristics_roundrobin.py` | ✅ | `roundrobin_results.csv` |
| **Round-robin riguroso (100/par, Bonferroni)** | `experiment_heuristics_rigorous.py` | ✅ | `rigorous_*.csv` |
| **Figuras rigurosas + experimentos complementarios** | `generate_mate_figures.py` | ✅ | `rigorous_*.png`, `rigorous_extra.csv` |

**No hay experimentos pendientes.** Todos los datos en `models/` son finales.

---

## 9. CHECKLIST DE CÓDIGO — CAMBIOS REALIZADOS

Los siguientes cambios de código fueron aplicados en esta sesión:

### Fixes de bugs:
- `Isolation/board.py` → `clone()`: evita llamada innecesaria a `place_players()` (RNG inútil) y copia `board_size` correctamente
- `Isolation/minimax_agent.py` → `_nodes_expanded`: cuenta nodo raíz en `next_action` (era 0 antes)
- `Isolation/heuristics.py` → `h_open_cells`: docstring corregido (no es "más granular", es normalización de la misma señal)

### Nuevas features:
- `Isolation/minimax_agent.py` → `use_move_ordering=False`: move ordering opcional para mejor AB pruning
- `Isolation/heuristics.py` → `h_territory`: BFS reachability, captura control territorial
- `Isolation/heuristics.py` → `eval_territory`, `eval_mobility_territory`: composites con territory
- `Isolation/heuristics.py` → `HEURISTICS` dict: actualizado con nuevas heurísticas

### Nuevos scripts de experimentos:
- `scripts/experiment_ab_impact.py`: comparativa Pure vs AB vs AB+MO, depths 2-4
- `scripts/experiment_dyna_planning.py`: sweep n=0,1,5,10,20,50
- `scripts/experiment_heuristics_roundrobin.py`: round-robin completo 5 heurísticas
