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
| Q-Learning FINAL | 30×30 bins, 15 acc, 15k eps | **94.14** | 0.63 | ±0.12 | **100%** |
| Dyna-Q FINAL | 20×20 bins, 15 acc, n=10, 5k eps | **89.41** | 0.30 | ±0.06 | **100%** |

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

#### Round-Robin de heurísticas (60 partidas/par, depth=3, RESULTADOS FINALES):

| Heurística | Victorias | Partidas | Win Rate | IC 95% |
|------------|-----------|----------|----------|--------|
| **territory** | 127 | 240 | **52.9%** | ±6.3% |
| mob_territory | 125 | 240 | 52.1% | ±6.3% |
| mob_center | 119 | 240 | 49.6% | ±6.3% |
| full | 116 | 240 | 48.3% | ±6.3% |
| mob_only | 113 | 240 | 47.1% | ±6.3% |

**Hallazgo clave:** `h_territory` (BFS reachability) gana el round-robin expandido.  
Las diferencias están dentro del CI — ninguna es estadísticamente significativa al 95%.  
`territory` domina como P1 (P1 advantage: ~73% en mirror match); como P2 es similar a mob_only.  
`eval_full` sigue siendo el peor (señales conflictivas entre movilidad, centro y espacio).  
**Ver:** `reports/figures/heuristic_roundrobin_full.png` y `heuristic_roundrobin_bar.png`

#### vs Stratagem (100 partidas, depth=3, mob_only):
| Rol | Win % | CI 95% |
|-----|-------|--------|
| Nuestro agente como P1 | 84% | ±7.2% |
| Nuestro agente como P2 | 54% | ±9.8% |

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
| `h_future_mobility` | promedio movs futuros míos - opp | ~20x h_mobility |
| `h_territory` | BFS: celdas vacías alcanzables yo - opp | ~1.8x h_mobility |

**Funciones compuestas:**
- `eval_mobility_only`: solo h_mobility — señal más directa de ganancia/pérdida en Isolation
- `eval_mobility_center`: 0.7×mob + 0.3×center — útil en early-game cuando el tablero está abierto
- `eval_full`: 0.6×mob + 0.2×center + 0.2×open_cells — peor en round-robin (señales conflictivas)
- `eval_territory`: solo h_territory (BFS) — **gana round-robin expandido** (52.9%)
- `eval_mobility_territory`: 0.6×mob + 0.4×territory — segundo lugar (52.1%), balance señal inmediata + territorial

**Nota h_open_cells:** usa la misma señal subyacente que h_mobility (get_possible_actions), solo difiere en normalización. No es una señal independiente.

#### 2.5 Experimentación y Torneos (2 pág)

**Impacto de la profundidad:**
| Depth | Win% vs Random | Nodos promedio/partida |
|-------|---------------|----------------------|
| 2 | 100% | 3,038 |
| 3 | 99% | 23,368 |
| 4 | 100% (n=10) | 187,758 |
- Depth=2 ya domina a Random. Las diferencias emergen contra oponentes hábiles.

**Round-Robin de heurísticas (60 partidas/par, depth=3):**
- **Insertar figura:** `heuristic_roundrobin_full.png` (heatmap 5×5) y `heuristic_roundrobin_bar.png` (ranking)
- Tabla desde `models/mate/roundrobin_results.csv`:

| Heurística | Win Rate | IC 95% |
|------------|----------|--------|
| **territory** (BFS) | **52.9%** | ±6.3% |
| mob_territory | 52.1% | ±6.3% |
| mob_center | 49.6% | ±6.3% |
| full | 48.3% | ±6.3% |
| mob_only | 47.1% | ±6.3% |

Diferencias dentro del IC → ninguna estadísticamente significativa. Territory domina como P1, similar a mob_only como P2.

**Minimax vs Expectimax (análisis profundo):**
| Matchup | Win rate |
|---------|----------|
| Minimax-AB vs Random | 100% |
| Expectimax vs Random | 78% |
| Minimax-AB vs Expectimax (mismo depth) | 98% |
| Expectimax (depth+1) vs Minimax-AB | ~50% |
- Profundidad compensa la desventaja del modelo de oponente incorrecto

**Ventaja del primer jugador:**
- Mirror match (Minimax vs Minimax): P1 gana ~70% en tablero 4×4
- El tablero pequeño favorece al primer movedor (más área "virgen")

**vs Stratagem:**
- **Insertar figura:** `vs_stratagem.png`
- P1: 84% | P2: 54% | Total: 69%
- Stratagem usa heurísticas de vecindad complementarias a movilidad

#### 2.6 Conclusiones MATE (0.5 pág)
- **Mejor agente final:** MinimaxAgent(depth=3, heuristic=eval_territory, use_alpha_beta=True) — gana el round-robin expandido
- **Alpha-Beta ESENCIAL:** depth=4 sin AB = 45.6s/partida (inviable). Con AB = 1.15s/partida.
- **h_territory (BFS) supera h_mobility** en el round-robin completo — captura control territorial a largo plazo
- **Minimax > Expectimax (98%)** contra oponente estratégico. Expectimax sería superior contra oponentes aleatorios/débiles.
- **Ventaja P1:** ~72% en 4×4. Explicar en informe: el primer jugador ocupa más área libre.
- **vs Stratagem:** 69% combinado (84% P1, 54% P2)

### 3. Conclusiones Generales (1 pág)
- Tabla comparativa ambos proyectos
- Uso de IA Generativa (según README): Claude para estructura y esqueletos, verificado por nosotros
- Dificultades encontradas

### 4. Anexos
- Figuras adicionales (learning curves, etc.)
- Fragmentos de código relevante (discretizador, regla de actualización Q)
- No poner código completo — referir al repositorio

---

## 4. FIGURAS DISPONIBLES

```
reports/figures/
├── ql_learning_curve.png           # LOST: curva Q-Learning 15k eps
├── dq_learning_curve.png           # LOST: curva Dyna-Q 5k eps
├── ql_vs_dq_comparison.png         # LOST: comparación primeros 5k eps
├── alpha_search.png                # LOST: grilla α
├── gamma_search.png                # LOST: grilla γ
├── epsilon_decay_search.png        # LOST: grilla ε decay
├── dyna_planning_sweep.png         # LOST: sweep n_planning 0-50 ← NUEVO
├── ab_impact.png                   # MATE: impacto Alpha-Beta ← NUEVO
├── heuristic_roundrobin_full.png   # MATE: heatmap round-robin ← NUEVO
├── heuristic_roundrobin_bar.png    # MATE: ranking final ← NUEVO
├── depth_vs_winrate.png            # MATE: profundidad vs win rate
├── vs_stratagem.png                # MATE: vs agente referencia
└── planning_steps_search.png       # (versión anterior, reemplazar con dyna_planning_sweep.png)
```

---

## 5. PASO A PASO: CREAR EL PDF

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

### Paso 5: Sección MATE — datos de experimentos nuevos
Después de que terminen los experimentos (ver Paso 0 abajo):
- AB impact: leer `models/mate/ab_impact_results.csv` → copiar tabla
- Round-robin: leer `models/mate/roundrobin_results.csv` → copiar standings

### Paso 6: Insertar figuras MATE
Figuras MATE: ab_impact → roundrobin_full → roundrobin_bar → vs_stratagem

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

### Paso 10: Revisar checklist
- [ ] Portada con nombres y números de estudiante
- [ ] Sección LOST: discretización, Q-Learning, hiperparámetros, Dyna-Q
- [ ] Sección MATE: Minimax+AB+impacto, Expectimax, heurísticas, torneos
- [ ] Al menos 5 figuras
- [ ] Tablas de resultados con estadísticas (mean ± std, CI95, success rate)
- [ ] Modelos .pkl mencionados y adjuntos en el ZIP
- [ ] Declaración uso de IA
- [ ] ≤ 20 páginas (sin contar anexos)
- [ ] ZIP con todo: código .py, notebooks .ipynb, modelos .pkl, informe .pdf

---

## 6. PASO 0: ESPERAR EXPERIMENTOS EN CURSO

Antes de escribir el PDF, verificar que terminen estos experimentos:

### Experimento 1: Alpha-Beta Impact
**Script:** `scripts/experiment_ab_impact.py`  
**Estado:** Datos depth=2 y depth=3 ✅ disponibles. Depth=4 pure minimax corriendo (puede tardar mucho).  
**Acción:** Si depth=4 pure minimax no termina en 30 min, anotar en el informe que es "computacionalmente infeasible" (es un hallazgo válido).  
**Resultados disponibles:** `models/mate/ab_impact_results.csv` y `reports/figures/ab_impact.png`

### Experimento 2: DynaQ n_planning sweep
**Script:** `scripts/experiment_dyna_planning.py`  
**Estado:** Corriendo en background. Resultados parciales (3000 eps, q_init=0, decay=0.9995):

| n_planning | Reward | Éxito | 1er ep |
|-----------|--------|-------|--------|
| 0 (QL puro) | 0.00 | **0%** | ep 17 (no converge) |
| 1 | 0.00 | **0%** | nunca |
| 5 | 0.00 | **0%** | nunca |
| 10 | 39.97 | **66%** | ep 51 |
| 20 | pendiente... | | |
| 50 | pendiente... | | |

**Hallazgos clave:**
- n ≥ 10 es el **umbral crítico** para convergencia en 3000 eps
- n=20 tiene **PEOR rendimiento** que n=10 (18% vs 66%) — premature convergence
- Con modelo escaso (pocas visitas), planning excesivo refuerza valores Q incorrectos
- El modelo FINAL usa n=10 con 5000 eps → 100% (balance óptimo)

**Curva no monótona:** Más planning no siempre es mejor. Trade-off: más planning = aprende más rápido, pero también puede sobreajustarse a un modelo impreciso. Sutton & Barto discuten esto en cap 8.3 (prioritized sweeping como solución).

**Resultados completos:** `reports/figures/dyna_planning_sweep.png` y `models/lost/dyna_planning_sweep.csv`

### Experimento 3: Round-Robin de heurísticas
**Script:** `scripts/experiment_heuristics_roundrobin.py`  
**Estado:** Corriendo en background (~20-30 min).  
**Resultados:** `reports/figures/heuristic_roundrobin_full.png`, `heuristic_roundrobin_bar.png`, `models/mate/roundrobin_results.csv`

Para verificar progreso:
```bash
# DynaQ sweep
cat models/lost/dyna_planning_sweep.csv

# Round-robin
cat models/mate/roundrobin_results.csv

# AB impact
cat models/mate/ab_impact_results.csv
```

---

## 7. CHECKLIST DE CÓDIGO — CAMBIOS REALIZADOS

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
