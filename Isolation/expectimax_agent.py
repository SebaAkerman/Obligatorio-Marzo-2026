import math
from typing import Callable

from agent import Agent
from board import Board
from heuristics import eval_mobility_only


INF = math.inf


class ExpectimaxAgent(Agent):
    # asume que el rival juega al azar (uniforme) en vez de jugar óptimo como en minimax

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
        return self.heuristic(board, self.player)

    def _max_value(self, board: Board, depth: int) -> float:
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

    def _chance_value(self, board: Board, depth: int) -> float:
        # nodo de azar: promedia el valor sobre todas las jugadas del rival,
        # asumiendo que elige cada una con la misma probabilidad
        self._nodes_expanded += 1
        opponent = self.player % 2 + 1
        is_end, winner = board.is_end(opponent)

        if is_end:
            return INF if winner == self.player else -INF

        if depth >= self.max_depth:
            return self.heuristic(board, self.player)

        possible_actions = board.get_possible_actions(opponent)

        if not possible_actions:
            return INF  # el rival se quedó sin movimientos, ganamos

        prob = 1.0 / len(possible_actions)
        expected_value = 0.0

        for action in possible_actions:
            child = board.clone()
            child.play(action, opponent)
            expected_value += prob * self._max_value(child, depth + 1)

        return expected_value
