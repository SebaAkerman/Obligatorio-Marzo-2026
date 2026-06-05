"""
Agente Minimax con Alpha-Beta Pruning para el juego Isolation.

Referencia: Russell & Norvig — Figura 5.7 (Alpha-Beta Search)
            MiniMax.pdf y MiniMax_1.pdf del curso (Yovine, ORT Uruguay)

Minimax asume que ambos jugadores juegan de forma óptima:
  - Max busca maximizar la utilidad (nuestro agente)
  - Min busca minimizar la utilidad (el oponente)

Alpha-Beta Pruning elimina ramas que no pueden afectar la decisión:
  - α: mejor valor que Max puede garantizarse (inicia en -∞)
  - β: mejor valor que Min puede garantizarse (inicia en +∞)
  - Si en un nodo Max tenemos valor ≥ β → poda (el Min padre no elegiría aquí)
  - Si en un nodo Min tenemos valor ≤ α → poda (el Max padre no elegiría aquí)

Move Ordering (opcional):
  Ordenar acciones por valor heurístico antes de expandirlas acelera la poda:
  el mejor movimiento se evalúa primero → α/β se actualizan más rápido → más ramas podadas.
  Costo: una llamada adicional a la heurística por acción en cada nodo.
"""

import math
from typing import Callable

from agent import Agent
from board import Board
from heuristics import eval_mobility_only


INF = math.inf


class MinimaxAgent(Agent):
    """
    Agente que decide usando Minimax con Alpha-Beta Pruning.

    Parámetros
    ----------
    player : int
        Número de jugador (1 o 2).
    max_depth : int
        Profundidad máxima del árbol de búsqueda.
    heuristic : Callable
        Función de evaluación h(board, player) → float.
    use_alpha_beta : bool
        Si True, activa Alpha-Beta Pruning. Si False, Minimax puro.
    use_move_ordering : bool
        Si True, ordena las acciones por valor heurístico antes de expandir.
        Mejora significativamente la eficiencia de la poda Alpha-Beta.
    """

    def __init__(
        self,
        player: int,
        max_depth: int = 3,
        heuristic: Callable = eval_mobility_only,
        use_alpha_beta: bool = True,
        use_move_ordering: bool = False,
    ) -> None:
        super().__init__(player)
        self.max_depth = max_depth
        self.heuristic = heuristic
        self.use_alpha_beta = use_alpha_beta
        self.use_move_ordering = use_move_ordering
        self._nodes_expanded = 0

    def next_action(self, obs: Board):
        """
        Selecciona la mejor acción usando Minimax (con o sin Alpha-Beta).

        Parámetros
        ----------
        obs : Board
            Estado actual del tablero.

        Devuelve
        --------
        best_action : tuple
            La acción (dirección, celda_a_destruir) óptima según Minimax.
        """
        self._nodes_expanded = 1  # nodo raíz
        best_action = None
        best_value = -INF
        alpha = -INF

        possible_actions = self._ordered_actions(obs, self.player, maximize=True)

        if not possible_actions:
            return None  # sin movimientos → perdimos

        for action in possible_actions:
            next_board = obs.clone()
            next_board.play(action, self.player)

            if self.use_alpha_beta:
                value = self._min_value(next_board, depth=1, alpha=alpha, beta=INF)
            else:
                value = self._min_value_pure(next_board, depth=1)

            if value > best_value:
                best_value = value
                best_action = action

            if self.use_alpha_beta:
                alpha = max(alpha, best_value)

        return best_action

    def heuristic_utility(self, board: Board) -> float:
        """Interfaz requerida por la clase abstracta Agent."""
        return self.heuristic(board, self.player)

    # ------------------------------------------------------------------
    # Move Ordering
    # ------------------------------------------------------------------

    def _ordered_actions(self, board: Board, player: int, maximize: bool = True) -> list:
        """
        Devuelve las acciones ordenadas por valor heurístico.
        Si use_move_ordering=False, devuelve el orden original.
        """
        actions = board.get_possible_actions(player)
        if not self.use_move_ordering or len(actions) <= 1:
            return actions

        scored = []
        for action in actions:
            child = board.clone()
            child.play(action, player)
            score = self.heuristic(child, self.player)
            scored.append((score, action))

        scored.sort(key=lambda x: x[0], reverse=maximize)
        return [a for _, a in scored]

    # ------------------------------------------------------------------
    # Minimax con Alpha-Beta Pruning
    # ------------------------------------------------------------------

    def _max_value(self, board: Board, depth: int, alpha: float, beta: float) -> float:
        """Nodo Max en el árbol Minimax."""
        self._nodes_expanded += 1
        is_end, winner = board.is_end(self.player)

        if is_end:
            return INF if winner == self.player else -INF

        if depth >= self.max_depth:
            return self.heuristic(board, self.player)

        value = -INF
        for action in self._ordered_actions(board, self.player, maximize=True):
            child = board.clone()
            child.play(action, self.player)
            value = max(value, self._min_value(child, depth + 1, alpha, beta))
            if value >= beta:
                return value  # poda β
            alpha = max(alpha, value)

        return value

    def _min_value(self, board: Board, depth: int, alpha: float, beta: float) -> float:
        """Nodo Min en el árbol Minimax."""
        self._nodes_expanded += 1
        opponent = self.player % 2 + 1
        is_end, winner = board.is_end(opponent)

        if is_end:
            return INF if winner == self.player else -INF

        if depth >= self.max_depth:
            return self.heuristic(board, self.player)

        value = INF
        for action in self._ordered_actions(board, opponent, maximize=False):
            child = board.clone()
            child.play(action, opponent)
            value = min(value, self._max_value(child, depth + 1, alpha, beta))
            if value <= alpha:
                return value  # poda α
            beta = min(beta, value)

        return value

    # ------------------------------------------------------------------
    # Minimax puro (sin poda) — para comparación y análisis
    # ------------------------------------------------------------------

    def _max_value_pure(self, board: Board, depth: int) -> float:
        self._nodes_expanded += 1
        is_end, winner = board.is_end(self.player)
        if is_end:
            return INF if winner == self.player else -INF
        if depth >= self.max_depth:
            return self.heuristic(board, self.player)

        value = -INF
        for action in board.get_possible_actions(self.player):
            child = board.clone()
            child.play(action, self.player)
            value = max(value, self._min_value_pure(child, depth + 1))
        return value

    def _min_value_pure(self, board: Board, depth: int) -> float:
        self._nodes_expanded += 1
        opponent = self.player % 2 + 1
        is_end, winner = board.is_end(opponent)
        if is_end:
            return INF if winner == self.player else -INF
        if depth >= self.max_depth:
            return self.heuristic(board, self.player)

        value = INF
        for action in board.get_possible_actions(opponent):
            child = board.clone()
            child.play(action, opponent)
            value = min(value, self._max_value_pure(child, depth + 1))
        return value
