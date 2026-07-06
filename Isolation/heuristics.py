from board import Board


def h_mobility(board: Board, player: int) -> float:
    """Diferencia de movimientos disponibles entre los dos jugadores."""
    opponent = player % 2 + 1
    my_moves = len(board.get_possible_actions(player))
    opp_moves = len(board.get_possible_actions(opponent))
    return float(my_moves - opp_moves)


def h_open_cells(board: Board, player: int) -> float:
    # mismo criterio que h_mobility pero normalizado a [-1, 1], para que no
    # dependa tanto de cuántas celdas libres queden en el tablero
    opponent = player % 2 + 1
    my_actions = len(board.get_possible_actions(player))
    opp_actions = len(board.get_possible_actions(opponent))
    total = my_actions + opp_actions
    if total == 0:
        return 0.0
    return float((my_actions - opp_actions) / total)


def h_center_proximity(board: Board, player: int) -> float:
    # estar cerca del centro suele dar más opciones de movimiento, así que
    # comparamos distancia Manhattan al centro contra la del rival
    opponent = player % 2 + 1
    pos_player = board.find_player_position(player)
    pos_opp = board.find_player_position(opponent)

    if pos_player is None or pos_opp is None:
        return 0.0

    rows, cols = board.board_size
    center = ((rows - 1) / 2, (cols - 1) / 2)

    dist_player = abs(pos_player[0] - center[0]) + abs(pos_player[1] - center[1])
    dist_opp = abs(pos_opp[0] - center[0]) + abs(pos_opp[1] - center[1])

    max_dist = (rows - 1) + (cols - 1)
    return float((dist_opp - dist_player) / max_dist)


def h_aggressive(board: Board, player: int) -> float:
    # ignora al rival, solo mira la movilidad propia
    my_moves = len(board.get_possible_actions(player))
    return float(my_moves)


def eval_mobility_only(board: Board, player: int) -> float:
    return h_mobility(board, player)


def eval_mobility_center(board: Board, player: int, w1: float = 0.7, w2: float = 0.3) -> float:
    return w1 * h_mobility(board, player) + w2 * h_center_proximity(board, player)


def eval_full(
    board: Board,
    player: int,
    w_mobility: float = 0.6,
    w_center: float = 0.2,
    w_space: float = 0.2,
) -> float:
    # heurística principal, combina movilidad + centro + espacio libre
    mob = h_mobility(board, player)
    cen = h_center_proximity(board, player)
    space = h_open_cells(board, player)
    return w_mobility * mob + w_center * cen + w_space * space


def h_future_mobility(board: Board, player: int) -> float:
    # mira un paso adelante: promedio de movimientos que quedan disponibles
    # después de cada jugada posible propia/rival. Más informativa que
    # h_mobility pero bastante más lenta (clona y juega cada acción)
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
    return h_future_mobility(board, player)


def h_territory(board: Board, player: int) -> float:
    # BFS desde cada jugador para contar celdas vacías alcanzables. Captura
    # control de territorio y no solo movilidad inmediata, pero es O(n^2)
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
    return h_territory(board, player)


def eval_mobility_territory(board: Board, player: int,
                             w_mob: float = 0.6, w_ter: float = 0.4) -> float:
    return w_mob * h_mobility(board, player) + w_ter * h_territory(board, player)


HEURISTICS = {
    "mobility_only": eval_mobility_only,
    "mobility_center": eval_mobility_center,
    "full": eval_full,
    "future_mobility": eval_future_mobility_only,
    "territory": eval_territory,
    "mobility_territory": eval_mobility_territory,
}
