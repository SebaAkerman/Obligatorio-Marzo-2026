import math
from typing import Callable

from agent import Agent
from board import Board
from heuristics import eval_mobility_only


INF = math.inf


class MinimaxAgent(Agent):
    # Minimax con poda alpha-beta opcional y ordenamiento de movimientos opcional.
    # use_move_ordering ordena las jugadas por heurística antes de expandir,
    # lo que ayuda bastante a que la poda corte antes.

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
        self._nodes_expanded = 1
        best_action = None
        best_value = -INF
        alpha = -INF

        possible_actions = self._ordered_actions(obs, self.player, maximize=True)

        if not possible_actions:
            return None

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
        return self.heuristic(board, self.player)

    def _ordered_actions(self, board: Board, player: int, maximize: bool = True) -> list:
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

    def _max_value(self, board: Board, depth: int, alpha: float, beta: float) -> float:
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
                return value  # poda beta
            alpha = max(alpha, value)

        return value

    def _min_value(self, board: Board, depth: int, alpha: float, beta: float) -> float:
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
                return value  # poda alfa
            beta = min(beta, value)

        return value

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
