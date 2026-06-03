"""
Agente Expectimax para el juego Isolation.

Referencia: Russell & Norvig — cap. 5.5 (Games with Chance)
            MiniMax.pdf del curso (Yovine, ORT Uruguay)

Expectimax difiere de Minimax en los nodos del oponente:
  - En lugar de asumir juego óptimo del oponente (Min),
    se calcula la ESPERANZA de los valores de los hijos.
  
  Expectimax(s) =
    U(s)                                   si esFinal(s)
    max_{a} Expectimax(suc(s,a))           si jugador(s) = Max
    Σ_a P(a) · Expectimax(suc(s,a))       si jugador(s) = Chance (oponente)

Por defecto el oponente se modela con distribución uniforme sobre
sus acciones posibles (P(a) = 1 / |acciones|).

Cuándo usar Expectimax vs Minimax:
  - Minimax: el oponente juega de forma óptima (conservador).
  - Expectimax: el oponente juega aleatoriamente o de forma subóptima 
    (más riesgoso pero potencialmente mejor recompensa promedio).
"""

import math
from typing import Callable

from agent import Agent
from board import Board
from heuristics import eval_mobility_only


INF = math.inf


class ExpectimaxAgent(Agent):
    """
    Agente que decide usando Expectimax.

    Parámetros
    ----------
    player : int
        Número de jugador (1 o 2).
    max_depth : int
        Profundidad máxima del árbol.
    heuristic : Callable
        Función de evaluación h(board, player) → float.
    """

    def __init__(
        self,
        player: int,
        max_depth: int = 3,
        heuristic: Callable = eval_mobility_only,
    ) -> None:
        super().__init__(player)
        self.max_depth = max_depth
        self.heuristic = heuristic
        self._nodes_expanded = 0

    def next_action(self, obs: Board):
        """
        Selecciona la mejor acción usando Expectimax.

        Parámetros
        ----------
        obs : Board
            Estado actual del tablero.

        Devuelve
        --------
        best_action : tuple | None
            La mejor acción según Expectimax, o None si no hay movimientos.
        """
        self._nodes_expanded = 0
        best_action = None
        best_value = -INF

        possible_actions = obs.get_possible_actions(self.player)

        if not possible_actions:
            return None

        for action in possible_actions:
            next_board = obs.clone()
            next_board.play(action, self.player)
            value = self._chance_value(next_board, depth=1)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def heuristic_utility(self, board: Board) -> float:
        """Interfaz requerida por la clase abstracta Agent."""
        return self.heuristic(board, self.player)

    # ------------------------------------------------------------------
    # Nodo Max (nuestro agente)
    # ------------------------------------------------------------------

    def _max_value(self, board: Board, depth: int) -> float:
        """Nodo Max: el agente elige la acción que maximiza la utilidad."""
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
            value = max(value, self._chance_value(child, depth + 1))

        return value

    # ------------------------------------------------------------------
    # Nodo Chance (oponente modelado como aleatorio)
    # ------------------------------------------------------------------

    def _chance_value(self, board: Board, depth: int) -> float:
        """
        Nodo Chance: el oponente elige una acción con distribución uniforme.
        Devuelve la esperanza del valor sobre todas las acciones posibles.
        """
        self._nodes_expanded += 1
        opponent = self.player % 2 + 1
        is_end, winner = board.is_end(opponent)

        if is_end:
            return INF if winner == self.player else -INF

        if depth >= self.max_depth:
            return self.heuristic(board, self.player)

        possible_actions = board.get_possible_actions(opponent)

        if not possible_actions:
            return -INF  # Oponente sin movimientos → ganamos

        # Distribución uniforme sobre las acciones del oponente
        prob = 1.0 / len(possible_actions)
        expected_value = 0.0

        for action in possible_actions:
            child = board.clone()
            child.play(action, opponent)
            expected_value += prob * self._max_value(child, depth + 1)

        return expected_value
