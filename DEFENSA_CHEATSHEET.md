# Cheat-sheet de Defensa — Obligatorio IA 2026 (LOST + MATE)

Respuestas basadas en los datos reales del proyecto. Formato defensa 2º parcial (pregunta oral).
Integrantes: Akerman (282163), Kelmanzon (282212). Agente final LOST y MATE ya validados con nuestros modelos.

> ⭐ **ACTUALIZACIÓN — corrida de alta potencia (1500 partidas/par, ~8h).** Escalar el tamaño de muestra **cambió 2 conclusiones** (contalo como fortaleza: "validamos con más datos y corregimos"):
> 1. **Agente final ahora es depth=5** (no d4). d5 vence a d4 el 63% (p≈10⁻⁵) y sube contra Stratagem de 69.4%→**76.7%**. depth=6 no aporta (50%) → techo en d5.
> 2. **Dyna-Q necesita q_init=1.0** (multi-seed 5 semillas): con q0 solo 2/5 semillas funcionan; con q1, 3/5 al 100% y 5/5 usables. El DQ FINAL entregado se re-entrenó con q1+semilla fija (91.9/100%, reproducible).
> 3. Heurísticas: con 1500 partidas las 4 mejores siguen empatadas (~50-51%); solo `mobility_territory` quedó peor (47%). `mobility_only` sigue siendo la elección (empata-mejor + más simple).
>
> **Regla de oro en la defensa:** decí el **qué**, el **número** y el **porqué**. Ej: "usamos γ=0.99 (número) porque con γ=0.90 la recompensa de la meta descontada vale 0.9⁵⁰⁰≈10⁻²³ y es invisible para el TD-update (porqué)."

---

## PARTE 1 — PROYECTO LOST (MountainCarContinuous + Q-Learning)

### ¿Qué algoritmo usaron y por qué?
**Q-Learning tabular**, off-policy, control TD, con política de exploración ε-greedy. Regla de actualización (Sutton & Barto ec. 6.8):

```
Q(s,a) ← Q(s,a) + α·[ r + γ·max_a' Q(s',a') − Q(s,a) ]
```

Como el ambiente es continuo (obs y acción), lo **discretizamos** para poder usar una tabla Q. Elegimos Q-Learning porque lo pide la letra y porque es off-policy: aprende la política óptima independientemente de la política de exploración.

### Rol de cada parámetro, cómo lo hicieron variar y por qué
| Param | Valor final | Rol | Qué probamos / por qué |
|---|---|---|---|
| **α** (learning rate) | **0.2** | Cuánto pesa cada nueva experiencia en la actualización | Barrido {0.05,0.1,0.15,0.2,0.3,0.5}. α=0.05→66% (aprende lento); α=0.2→93.9/100% (óptimo); **α=0.3→0% (la Q-table diverge)**; α=0.5 inestable. |
| **γ** (descuento) | **0.99** | Cuánto valen las recompensas futuras | Barrido {0.90,0.95,0.99,0.999}. **γ≤0.95 → 0% éxito**: la meta está a ~500 pasos, y 0.9⁵⁰⁰≈10⁻²³ la hace invisible. γ=0.99 → 0.99⁵⁰⁰≈0.0066, suficiente. Es el hiperparámetro **más sensible** del proyecto. |
| **ε** (exploración) | 1.0 → 0.05, decay 0.9995 | Prob. de acción aleatoria (ε-greedy) | ε alto al inicio (explorar), decae para explotar. Barrido decay {0.999, 0.9995, 0.9997, 0.9999}: 0.9995 mejor (94.74). decay=0.9999 deja ε=0.61 al final → 84% (nunca explota). |
| **q_init** | **1.0 (optimista)** | Valor inicial de la tabla Q | **Clave.** Con q_init=0 el agente aprende "fuerza=0" y nunca llega a la meta (ver abajo). q_init=1.0 fuerza a explorar cada estado. |
| **bins** | 30×30 | Resolución de la discretización de estados | Barrido 10/20/30. 10×10 sufre aliasing; 30×30 mejor reward con espacio manejable. |
| **acciones** | 15 | Discretización de la fuerza continua | 5=control muy tosco; 15=buen balance; 20=redundante. |

### ¿Cuál fue el reward promedio con la mejor policy?
- **Q-Learning FINAL: 94.14 ± 0.63, 100% de éxito** (30×30 bins, 15 acc, 15k eps). Multi-seed: 93.02 ± 0.97 entre 3 semillas.
- **Dyna-Q FINAL: 91.9, 100% éxito** (20×20, 15 acc, n=10, 5k eps, **q_init=1.0** semilla fija).
- (Verificado al re-cargar los .pkl: QL ~94/100%, DQ ~91.9/100%.)

### Funciones de decaimiento de ε y α. ¿Cuál fue más efectiva?
- **ε: decaimiento multiplicativo** `ε ← max(ε_min, ε·decay)` con decay=0.9995, ε_min=0.05. Es exponencial. Probamos también decaimientos más lentos/rápidos; 0.9995 fue el mejor balance (más lento no baja ε lo suficiente para explotar; más rápido deja de explorar antes de encontrar la meta).
- **α: constante (0.2)**, no la decaímos. En un ambiente estacionario con tabla discreta, α constante convergió mejor que decaerla; decaer α demasiado rápido congela el aprendizaje antes de propagar el +100 de la meta.
- Más efectiva: el **decay exponencial de ε** fue el que más impacto tuvo junto con q_init y γ.

### ¿Cómo discretizaron el espacio de observaciones?
`Discretizer` (utils/discretization.py): grillas uniformes con `np.linspace` sobre los rangos del ambiente (pos ∈ [-1.2, 0.6] → 30 bins; vel ∈ [-0.07, 0.07] → 30 bins). `np.digitize(x, bins)` mapea la observación continua al índice de bin. Estado = `(x_bin, vel_bin)`. Tabla Q de forma `(31, 31, 15)`.

### Las acciones también son continuas — ¿cómo las discretizaron? Trade-off fino vs grueso
Fuerza continua ∈ [-1, 1] → `np.linspace(-1, 1, 15)` = 15 acciones discretas. El índice elegido por la política se mapea a la fuerza continua que espera `env.step()`.
- **Muy fino** (ej. 20+ acciones): control de fuerza preciso, pero el espacio (estados×acciones) crece → más exploración necesaria, convergencia más lenta, redundancia (acciones casi iguales).
- **Muy grueso** (ej. 5 acciones): aprende rápido pero pierde control fino del impulso — no puede modular la fuerza para generar el balanceo óptimo.
- **15 fue el punto óptimo**: precisión suficiente sin inflar el espacio.

### ¿Se enfocaron en eficiencia del entrenamiento o en el resultado? ¿Cuánto tardó?
Nos enfocamos en el **resultado** (100% de éxito, reward alto, validado con multi-seed), no en minimizar el tiempo. Aun así el entrenamiento es rápido: **QL FINAL ~5 min (15k eps), Dyna-Q ~4 min (5k eps)** en CPU.

### ¿Qué estrategia de exploración usaron? ¿Garantiza convergencia?
**ε-greedy con ε decreciente**, que se aproxima a **GLIE** (Greedy in the Limit with Infinite Exploration): si ε→0 lo suficientemente lento y cada par (s,a) se visita infinitas veces, Q-Learning converge a Q* con probabilidad 1. En la práctica usamos ε_min=0.05 (no exactamente 0) para mantener algo de exploración; combinado con **inicialización optimista** (q_init=1.0) que garantiza visitar todos los estados al menos una vez.

### ¿Por qué Q-Learning (off-policy) y no SARSA (on-policy)?
- **Q-Learning** actualiza hacia `max_a' Q(s',a')` — la mejor acción posible (política greedy target), aunque haya explorado con ε-greedy. Aprende **directamente la política óptima**.
- **SARSA** actualiza hacia `Q(s',a')` de la acción realmente tomada — aprende la política que incluye la exploración (más conservadora/segura).
- En MountainCar no hay penalización por "riesgo" durante el aprendizaje (no hay estados catastróficos que evitar), así que la agresividad de Q-Learning para converger al óptimo es preferible. Además la letra pide Q-Learning.

### ¿Cómo verificaron la convergencia? Criterio de corte
- **Curva de recompensa por episodio** + media móvil de 100 episodios (figura `ql_learning_curve.png`). Hitos: primer éxito ep **732**, media-100 ≥ 50 en ep **2368**, política estable desde ~ep 5000.
- **Criterio de corte:** entrenar 15k episodios y validar con `test_agent` en política greedy pura (ε=0): 100% de éxito y reward ~94. Confirmado además con **3 semillas** distintas (multiseed).

### Dyna-Q: rol del modelo, pasos de planning, comparación vs Q-Learning
- **Modelo aprendido:** `Model(s,a) → (r, s', done)`, determinista y tabular. Guarda cada transición real observada.
- **Planning:** por cada paso real hacemos **n=10** actualizaciones simuladas: se samplea un par (s,a) ya visitado, se recupera (r,s') del modelo y se aplica la misma regla de Q. Con n=0, Dyna-Q ≡ Q-Learning puro.
- **Comparación:** Dyna-Q converge en **5k eps (3× menos que QL)** porque el replay propaga el valor de la meta hacia atrás más rápido. Trade-off: cada episodio es más lento en cómputo (10 updates extra por paso), pero necesita mucha menos interacción real con el ambiente. Reward final levemente menor (91.9 vs 94.1) y mayor varianza entre semillas (sensibilidad al RNG en recompensa esparsa).
- **Sweep n_planning (hallazgo no obvio):** la curva es **no monótona** — n=10 óptimo (66% a 3k eps), pero n=20 cae a 18% y n=50 a 0%. Demasiada planificación sobre un modelo escaso refuerza valores Q incorrectos.
- **Limitación honesta:** Dyna-Q con q_init=0 es **sensible a la semilla** (necesita hallar la meta al menos una vez por exploración; con RNG desfavorable, 0% en 3 seeds). Con q_init=1.0 es más robusto (seeds 0/1→100%, seed2→70%, seed42→82%). Documentado en Sección 4 del informe.

---

## PARTE 2 — PROYECTO MATE (Isolation + Minimax/Expectimax)

### ¿Qué técnicas implementaron?
**Minimax con Alpha-Beta Pruning** (+ minimax puro para comparar, + move ordering opcional) y **Expectimax**. Agente final: `MinimaxAgent(depth=5, heuristic=eval_mobility_only, use_alpha_beta=True)`.

### Si el oponente no juega óptimo, ¿por qué Expectimax podría ser más apropiado que Minimax?
- **Minimax** asume el **peor caso** (oponente óptimo que minimiza nuestra utilidad) → conservador.
- **Expectimax** modela al oponente como **nodo de azar** (esperanza sobre sus acciones, distribución uniforme) → apropiado si el oponente es **aleatorio o subóptimo**, porque no "desperdicia" cuidándose de jugadas que el rival nunca haría.
- **Pero en nuestro caso Minimax gana:** el oponente real (Stratagem = Minimax d3, y otros Minimax) juega **casi óptimo**, así que la asunción de Minimax es correcta. Datos: **MM d3 vs EX d3 → MM gana 95.3% (p≈0)**. Expectimax solo fue mejor a **depth=2** (MM 41.7%) o contra un oponente genuinamente aleatorio.

### ¿Qué profundidad usaron y cómo la eligieron (trade-off tiempo vs calidad)?
**depth=5** (tras escalar la muestra a n=300). Elección por experimento:
- **Calidad:** d4 vence a d3 (61.2%, p≈0) y **d5 vence a d4 con 63% (p≈10⁻⁵, n=300)** → cada nivel hasta 5 aporta; la profundidad es el factor decisivo.
- **Tiempo:** d4 sin poda = **45.6 s/jugada (inviable)**; con Alpha-Beta d4 = 1.15 s, d5 ≈ **1.4 s/jugada (jugable)**.
- **Techo en d5:** **d6 NO mejora sobre d5** (50%, n=60, p=1.0) y contra Stratagem d5=76.7% vs d6=73.3%. Por eso el agente final es **depth=5**. (Con muestras chicas de 100 partidas la ventaja d5 no se distinguía del ruido; se reveló al escalar.)

### ¿Cómo impactó Alpha-Beta Pruning (nodos/tiempo antes vs después)?
Medido en 50 partidas por configuración:
| Depth | Minimax puro | Alpha-Beta | AB + Move Ordering |
|---|---|---|---|
| 2 | 3.608 n / 0.14 s | 795 n / 0.03 s (−78%) | 286 n / 0.14 s (−92%) |
| 3 | 91.153 n / 3.18 s | 8.752 n / 0.30 s (−90%) | 2.341 n / 0.44 s (−97%) |
| **4** | **1.586.170 n / 45.6 s** | **40.688 n / 1.15 s (−97%)** | 5.526 n / 1.94 s (−99.7% nodos) |

- Complejidad: O(b^d) → **O(b^(d/2))** en el mejor caso, **sin cambiar la jugada elegida** (AB devuelve el mismo resultado que Minimax puro).
- **Paradoja del Move Ordering:** reduce 99.7% los nodos pero es **más lento** (1.94 s vs 1.15 s) porque el costo de evaluar la heurística para ordenar (en cada nodo) supera el ahorro cuando la heurística es O(1). MO solo conviene con heurísticas caras (territory, future_mobility).

### ¿Qué funciones de evaluación usaron? ¿Cómo llegaron a ellas? ¿Cuál dio mejor?
5 heurísticas primitivas + composiciones ponderadas:
| Función | Fórmula | 
|---|---|
| `h_mobility` | mis_movimientos − movimientos_oponente |
| `h_center_proximity` | (dist_opp − dist_propia) / dist_máx (normalizada) |
| `h_open_cells` | (acc_propias − acc_opp) / total (misma señal que mobility, normalizada) |
| `h_territory` | BFS: celdas vacías alcanzables propias − del oponente |
| `h_future_mobility` | movilidad promedio tras 1 jugada, propia − opp (~20× más cara) |
| `eval_mobility_only` | h_mobility ← **elegida** |
| `eval_mobility_center` | 0.7·mobility + 0.3·center |
| `eval_full` | 0.6·mobility + 0.2·center + 0.2·open_cells |
| `eval_mobility_territory` | 0.6·mobility + 0.4·territory |

**Cómo llegamos:** partimos de la intuición del juego (quien se queda sin movimientos pierde → movilidad relativa es la señal más directa). Agregamos señales de posición (centro) y control (territorio BFS) y las combinamos con pesos. Luego hicimos un **experimento riguroso** para elegir objetivamente.

**Cuál dio mejor:** el round-robin escalado (**1500 partidas/par** + **Bonferroni** α=0.005) mostró que las **4 mejores heurísticas son estadísticamente equivalentes** (~50-51%) en 4×4 a depth=3; **solo `mobility_territory` quedó peor** (47%, p≈2×10⁻⁵). Por eso elegimos **`mobility_only`**: empata-mejor (51.1%), la más simple (una resta), la más rápida (O(1)). **La profundidad importa más que la heurística.**

### Si combinaron varias heurísticas ponderadas, ¿cómo determinaron los pesos?
Probamos varias combinaciones (0.7/0.3, 0.6/0.2/0.2, 0.6/0.4) y las comparamos en el round-robin. Como **ninguna combinación superó estadísticamente a `mobility_only`**, aplicamos navaja de Occam: la más simple. Los pesos exactos resultaron irrelevantes en este tablero pequeño porque a mayor profundidad el lookahead compensa la calidad de la evaluación de las hojas.

### ¿Contra qué oponentes testearon y cómo midieron el desempeño?
- **RandomAgent:** 90.5% (n=200, dominancia básica).
- **Stratagem** (agente de referencia de la cátedra, un Minimax d3): **76.7%** a depth=5 (n=300, p≈0, CI Wilson [71.6%, 81.1%]); a depth=4 era 69.4%.
- **Sí mismo / mirror match:** a depth=5 P1 gana **100%** (a depth=4 era 73%) → **ventaja estructural del primer jugador** en 4×4, total con más lookahead.
- **Métrica:** win-rate **balanceado** (50% de partidas como P1, 50% como P2, para neutralizar la ventaja de P1) + **test binomial** de dos colas + **IC Wilson 95%** + **corrección de Bonferroni** en comparaciones múltiples.

---

## PREGUNTAS EXTRA PROBABLES (fundamentos teóricos)

**Propiedades de Minimax:** completo (si el árbol es finito), **óptimo si el oponente juega óptimo**, complejidad temporal O(b^d), espacial O(b·d) (DFS). Alpha-Beta no cambia el resultado, solo la eficiencia.

**¿Qué es la función de evaluación/heurística?** Estima la utilidad de un estado **no terminal** cuando no se puede llegar al final del árbol. Debe ser (1) rápida (se llama en cada hoja), (2) **ordenar correctamente** los estados (que un estado mejor tenga mayor valor). En Isolation: movilidad relativa.

**¿Por qué tabular y no DQN/aproximación?** El ambiente es discretizable a un espacio manejable (~14k estados-acción); tabular converge garantizado y es interpretable. La letra pide Q-Learning tabular.

**¿Qué es off-policy?** La política que se **evalúa/mejora** (greedy) difiere de la que se **usa para actuar** (ε-greedy). Q-Learning es off-policy; SARSA on-policy.

**Manejo de estados terminales en Minimax:** si un jugador no tiene movimientos, pierde → utilidad +∞ (ganamos) o −∞ (perdemos), sin llamar a la heurística.

**¿Expectimax con poda?** No aplica Alpha-Beta clásico en nodos de azar (no hay min/max que podar), por eso Expectimax es más caro y lo limitamos a profundidades menores.

**Diferencia Monte Carlo vs TD vs Q-Learning (por si preguntan el marco):** MC actualiza al final del episodio con el retorno real; TD(0) actualiza cada paso con bootstrap `r+γV(s')`; Q-Learning es TD(0) de control, off-policy, sobre Q(s,a).

---

## TIEMPOS DE CÓMPUTO — ¿cuánto tardó y por qué? (pregunta típica de parcial)

| Tarea | Tiempo | Por qué |
|---|---|---|
| **QL FINAL** (30×30, 15k eps) | **~5 min** | Tabular = updates O(1); el costo es la interacción con el env (15k eps × ~200-500 pasos). Rápido porque no hay red neuronal ni backprop. |
| **Dyna-Q FINAL** (20×20, 5k eps, n=10) | **~4 min** | Menos episodios (5k vs 15k) pero **cada paso hace 11 updates** (1 real + 10 planning) → cada episodio es más lento; se compensa con 3× menos episodios. |
| Cada config de discretización (5k eps) | ~90 s | Igual que QL; configs de más bins tardan un poco más por la tabla más grande. |
| **Minimax por jugada, depth=3 (AB)** | **~0.3 s** | Árbol podado ~b^(d/2). |
| **Minimax por jugada, depth=4 (AB)** | **~1.15 s** | Jugable. |
| Minimax por jugada, depth=4 **SIN** AB | **45.6 s** | Inviable: explora ~1.586.000 nodos (b^d completo, b~96). **Por eso Alpha-Beta es obligatorio.** |
| Experimento AB impact (depth 2-4, 50 partidas c/u) | ~40 min | Dominado por el depth-4 puro (45.6 s × 50 partidas). |
| Experimento riguroso completo (fases 1-4, ~500 partidas depth 3-4) | **~74 min** | Muchas partidas × muchas jugadas × depth alto. Es el más caro. |

**Frase para el parcial:** "Q-Learning es rápido (~5 min) porque es tabular: cada actualización es O(1), sin backpropagation. Dyna-Q tarda parecido con 3× menos episodios porque hace 10 pasos de planning extra por paso real. En MATE el cuello de botella es el factor de ramificación (~96); a depth 4 sin poda son ~1.6M nodos y 45 s por jugada — Alpha-Beta lo baja a 1.15 s (−97%), y por eso es imprescindible."

**Eficiencia vs resultado:** priorizamos el **resultado** (100% éxito validado multi-seed), no minimizar tiempo — aunque igual quedó rápido. En MATE elegimos depth=5 (el techo útil: d6 ya no mejora) porque cada nivel hasta 5 sube significativamente vs Stratagem (69.4%→76.7%) y a ~1.4 s/jugada sigue siendo jugable.

---

## PUNTOS DÉBILES A TENER PREPARADOS (por si los atacan)
1. **Dyna-Q q_init=0 no es reproducible entre semillas** → respuesta: lo documentamos honestamente (Sección 4); el modelo entregado funciona (100% al cargar); con q_init=1.0 o ≥10k eps es robusto; es una propiedad del reward esparso, no un bug.
2. **Heurísticas "no significativas"** → no es debilidad: es un hallazgo con respaldo estadístico (Bonferroni). Justifica elegir la más simple.
3. **Sweep n_planning con q_init=0 mayormente 0%** → es el mismo fenómeno del reward esparso; el punto n=10 muestra el balance. La conclusión (no monotonía) es válida.
4. **Ventaja de P1 (73% a d4, 100% a d5)** → propiedad del tablero 4×4 que crece con la profundidad, no de los agentes; se mitiga balanceando roles (50% P1 / 50% P2) en todos los torneos.
