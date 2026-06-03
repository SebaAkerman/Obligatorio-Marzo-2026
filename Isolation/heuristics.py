"""
Funciones de evaluación heurística para el juego Isolation.

Una función de evaluación h(board, player) estima la utilidad de un
estado no terminal desde la perspectiva de `player`.

Referencia: Russell & Norvig — cap. 5 (Juegos de Suma Cero)
            MiniMax.pdf y MiniMax_1.pdf del curso

Criterios de diseño:
  - h debe ser rápida de calcular (se llama en cada nodo del árbol).
  - h debe ser informativa: capturar ventajas reales del jugador.
  - Experimentar con combinaciones ponderadas de distintas señales.

Convención de signos:
  Valores positivos → favorable para `player` (Max)
  Valores negativos → favorable para el oponente (Min)
"""

from board import Board


# ---------------------------------------------------------------------------
# Heurísticas primitivas
# ---------------------------------------------------------------------------

def h_mobility(board: Board, player: int) -> float:
    """
    Diferencia de movilidad: movimientos propios − movimientos del oponente.
    
    Razonamiento: tener más opciones de movimiento es ventajoso; 
    el jugador sin movimientos pierde.
    """
    opponent = player % 2 + 1
    my_moves = len(board.get_possible_actions(player))
    opp_moves = len(board.get_possible_actions(opponent))
    return float(my_moves - opp_moves)


def h_open_cells(board: Board, player: int) -> float:
    """
    Diferencia de acciones disponibles (ancho del espacio de acción).

    Razonamiento: más acciones disponibles = más libertad táctica.
    A diferencia de h_mobility (movimientos posibles), esto cuenta
    combinaciones (dirección × celda_a_destruir), más granular.
    """
    opponent = player % 2 + 1
    my_actions = len(board.get_possible_actions(player))
    opp_actions = len(board.get_possible_actions(opponent))
    total = my_actions + opp_actions
    if total == 0:
        return 0.0
    return float((my_actions - opp_actions) / total)


def h_center_proximity(board: Board, player: int) -> float:
    """
    Proximidad al centro del tablero.
    
    Estar en el centro ofrece más opciones de movimiento en promedio.
    Devuelve valor entre -1 y 1: positivo si el jugador está más 
    cerca al centro que el oponente.
    """
    opponent = player % 2 + 1
    pos_player = board.find_player_position(player)
    pos_opp = board.find_player_position(opponent)

    if pos_player is None or pos_opp is None:
        return 0.0

    rows, cols = board.board_size
    center = ((rows - 1) / 2, (cols - 1) / 2)

    dist_player = abs(pos_player[0] - center[0]) + abs(pos_player[1] - center[1])
    dist_opp = abs(pos_opp[0] - center[0]) + abs(pos_opp[1] - center[1])

    # Normalizar por la distancia máxima posible
    max_dist = (rows - 1) + (cols - 1)
    return float((dist_opp - dist_player) / max_dist)


def h_aggressive(board: Board, player: int) -> float:
    """
    Movilidad ponderando más los movimientos propios.
    
    Variante más agresiva: penaliza menos al oponente, prioriza
    maximizar las propias opciones.
    """
    my_moves = len(board.get_possible_actions(player))
    return float(my_moves)


# ---------------------------------------------------------------------------
# Funciones de evaluación compuestas
# ---------------------------------------------------------------------------

def eval_mobility_only(board: Board, player: int) -> float:
    """Evalúa únicamente por diferencia de movilidad."""
    return h_mobility(board, player)


def eval_mobility_center(board: Board, player: int, w1: float = 0.7, w2: float = 0.3) -> float:
    """
    Combinación ponderada de movilidad y proximidad al centro.
    
    Parámetros
    ----------
    w1 : float
        Peso de la diferencia de movilidad.
    w2 : float
        Peso de la proximidad al centro.
    """
    return w1 * h_mobility(board, player) + w2 * h_center_proximity(board, player)


def eval_full(
    board: Board,
    player: int,
    w_mobility: float = 0.6,
    w_center: float = 0.2,
    w_space: float = 0.2,
) -> float:
    """
    Función de evaluación completa: movilidad + centro + espacio libre.
    
    Esta es la función de evaluación principal a experimentar.
    Ajustar los pesos para optimizar el rendimiento del agente.
    """
    mob = h_mobility(board, player)
    cen = h_center_proximity(board, player)
    space = h_open_cells(board, player)

    return w_mobility * mob + w_center * cen + w_space * space


# ---------------------------------------------------------------------------
# Catálogo de funciones de evaluación para experimentación
# ---------------------------------------------------------------------------

HEURISTICS = {
    "mobility_only": eval_mobility_only,
    "mobility_center": eval_mobility_center,
    "full": eval_full,
}
