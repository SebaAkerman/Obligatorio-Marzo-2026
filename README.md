# Obligatorio IA 2026 — Red Destination™

**Materia:** Inteligencia Artificial — Universidad ORT Uruguay  


---

## Estructura del repositorio

```
Obligatorio-Marzo-2026/
│
├── MountainCarContinuous/         # Proyecto LOST — Q-Learning
│   ├── pyproject.toml             # Entorno Poetry (Python ~3.10)
│   ├── poetry.lock
│   ├── continuous_mountain_car.ipynb  # Notebook base (cátedra)
│   ├── q_learning_agent.py        # Agente Q-Learning tabular
│   ├── dyna_q_agent.py            # Agente Dyna-Q (Sutton & Barto caps. 8.1-8.2)
│   └── utils/
│       ├── __init__.py
│       └── discretization.py      # Discretización de obs/acciones
│
├── Isolation/                     # Proyecto MATE — Minimax/Expectimax
│   ├── pyproject.toml             # Entorno Poetry (Python ~3.10)
│   ├── poetry.lock
│   ├── agent.py                   # Clase abstracta Agent (cátedra)
│   ├── board.py                   # Lógica del tablero (cátedra)
│   ├── isolation_env.py           # Entorno Gymnasium (cátedra)
│   ├── play.py                    # Utilidad para ejecutar partidas (cátedra)
│   ├── input_agent.py             # Agente manual (cátedra)
│   ├── random_agent.py            # Agente aleatorio (cátedra)
│   ├── stratagem.py               # Agente de referencia (cátedra)
│   ├── isolation.ipynb            # Notebook base (cátedra)
│   ├── minimax_agent.py           # Minimax + Alpha-Beta Pruning
│   ├── expectimax_agent.py        # Expectimax
│   └── heuristics.py             # Funciones de evaluación heurística
│
├── models/
│   ├── lost/                      # Modelos .pkl entrenados (Q-Learning, Dyna-Q)
│   └── mate/                      # Resultados de torneos
│
├── reports/
│   └── figures/                   # Gráficos para el informe
│
├── scripts/
│   ├── train_lost.py              # CLI para entrenar agentes LOST
│   └── evaluate_mate.py           # CLI para torneos MATE
│
├── Obligatorio 2026 Marzo.pdf
└── README.md
```

---

## Proyectos

### Proyecto LOST — Learning-based Orientation and Steering for Traversal

**Ambiente:** `MountainCarContinuous-v0` (Gymnasium)  
**Técnica principal:** Q-Learning tabular con discretización  
**Investigación:** Dyna-Q (Sutton & Barto, caps. 8.1 y 8.2)

Tareas:
1. Discretizar observaciones y acciones — explorar distintas granularidades
2. Implementar Q-Learning con exploración ε-greedy
3. Búsqueda de hiperparámetros (α, γ, ε, bins, n_actions)
4. Implementar Dyna-Q y comparar con Q-Learning

### Proyecto MATE — Martian Adversarial Tactics Engine

**Ambiente:** Isolation (tablero 4×4)  
**Técnicas:** Minimax con Alpha-Beta Pruning + Expectimax

Tareas:
1. Implementar Minimax con Alpha-Beta Pruning y analizar su impacto
2. Implementar Expectimax
3. Diseñar y comparar funciones de evaluación heurística
4. Experimentación con torneos entre agentes

---

## Setup

```bash
# Clonar
git clone https://github.com/SebaAkerman/Obligatorio-Marzo-2026.git
cd Obligatorio-Marzo-2026

# Proyecto LOST
cd MountainCarContinuous
poetry install
poetry run jupyter notebook continuous_mountain_car.ipynb

# Proyecto MATE (otra terminal)
cd ../Isolation
poetry install
poetry run jupyter notebook isolation.ipynb
```

### Ejecutar scripts desde cada entorno

```bash
# Entrenar Q-Learning (desde MountainCarContinuous/)
cd MountainCarContinuous
poetry run python ../scripts/train_lost.py --agent qlearning --episodes 5000

# Entrenar Dyna-Q
poetry run python ../scripts/train_lost.py --agent dynaq --n-planning 10

# Torneo Minimax vs Random (desde Isolation/)
cd ../Isolation
poetry run python ../scripts/evaluate_mate.py --agent1 minimax --agent2 random --n-games 100

# Torneo Minimax vs Expectimax
poetry run python ../scripts/evaluate_mate.py --agent1 minimax --agent2 expectimax --depth 3
```

---

## Uso de IA Generativa

Este proyecto utilizó Claude (Anthropic) como apoyo en:
- Estructuración y organización del repositorio
- Generación de código base (esqueletos de clases y funciones)
- Revisión y depuración de implementaciones

Todo el contenido fue revisado, verificado y comprendido por los integrantes del equipo. Los errores presentes son de nuestra responsabilidad. 

---

## Integrantes

| Nombre | Nro. Estudiante |
|---|---|
| Sebastian Akerman | 282163 |
| Felipe Kelmanzon | 282212 |

**Dictado:** Ingeniería en Sistemas — Inteligencia Artificial