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
    Diferencia de movilidad normalizada.

    Nota: usa la misma señal que h_mobility (get_possible_actions) pero
    normalizada por el total de acciones disponibles → valor en [-1, 1].
    La normalización evita que el valor dependa de la escala absoluta
    de movilidad (tablero lleno vs vacío), haciendo la señal más estable
    entre etapas del juego.
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


def h_future_mobility(board: Board, player: int) -> float:
    """
    Movilidad futura esperada: promedio de movimientos disponibles
    tras cada posible acción propia, menos el promedio del oponente.

    Más informativa que h_mobility porque mira un paso adelante,
    pero ~20x más cara de calcular (usa clone+play para cada acción).
    """
    opponent = player % 2 + 1

    my_actions = board.get_possible_actions(player)
    if not my_actions:
        return float(-1e9)
    my_future = 0.0
    for action in my_actions:
        child = board.clone()
        child.play(action, player)
        my_future += len(child.get_possible_actions(player))
    my_future /= len(my_actions)

    opp_actions = board.get_possible_actions(opponent)
    if not opp_actions:
        return float(1e9)
    opp_future = 0.0
    for action in opp_actions:
        child = board.clone()
        child.play(action, opponent)
        opp_future += len(child.get_possible_actions(opponent))
    opp_future /= len(opp_actions)

    return float(my_future - opp_future)


def eval_future_mobility_only(board: Board, player: int) -> float:
    """Evalúa únicamente por movilidad futura esperada."""
    return h_future_mobility(board, player)


def h_territory(board: Board, player: int) -> float:
    """
    Territorio BFS: diferencia de celdas vacías alcanzables desde cada jugador.

    A diferencia de h_mobility (movimientos inmediatos), esta métrica usa BFS
    para calcular el área total de celdas vacías accesibles transitivamente
    desde la posición de cada jugador. Captura el control de territorio a largo
    plazo, no solo la movilidad inmediata.

    Costo: O(n²) por jugador (BFS sobre el tablero), más caro que h_mobility
    pero más informativa en tableros fragmentados.
    """
    from collections import deque

    def bfs_area(start_pos, grid, board_size):
        if start_pos is None:
            return 0
        visited = {start_pos}
        queue = deque([start_pos])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < board_size[0] and 0 <= nc < board_size[1]
                        and grid[nr, nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return len(visited)

    opponent = player % 2 + 1
    pos_p = board.find_player_position(player)
    pos_o = board.find_player_position(opponent)

    my_area = bfs_area(pos_p, board.grid, board.board_size)
    opp_area = bfs_area(pos_o, board.grid, board.board_size)
    return float(my_area - opp_area)


def eval_territory(board: Board, player: int) -> float:
    """Evalúa únicamente por territorio BFS."""
    return h_territory(board, player)


def eval_mobility_territory(board: Board, player: int,
                             w_mob: float = 0.6, w_ter: float = 0.4) -> float:
    """
    Combinación ponderada de movilidad inmediata y territorio BFS.
    Balancea reacción inmediata (mobility) con control territorial (BFS).
    """
    return w_mob * h_mobility(board, player) + w_ter * h_territory(board, player)


# ---------------------------------------------------------------------------
# Catálogo de funciones de evaluación para experimentación
# ---------------------------------------------------------------------------

HEURISTICS = {
    "mobility_only": eval_mobility_only,
    "mobility_center": eval_mobility_center,
    "full": eval_full,
    "future_mobility": eval_future_mobility_only,
    "territory": eval_territory,
    "mobility_territory": eval_mobility_territory,
}
