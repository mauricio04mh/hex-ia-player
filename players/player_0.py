from __future__ import annotations

"""
- It only generates legal moves.
- It chooses one of them uniformly at random.
- It has no tactics, no heuristic, and no search.

"""

import random

from board import HexBoard
from player import Player


Move = tuple[int, int]


class RandomPlayer(Player):
    """
    Very simple HEX player that always returns a legal random move.
    """

    def __init__(self, player_id: int, seed: int | None = None):
        """
        Initialize the random player.

        Args:
            player_id: The ID of this player (usually 1 or 2).
            seed: Optional seed for reproducible randomness during testing.
        """
        super().__init__(player_id)
        self.rng = random.Random(seed)

    def play(self, board: HexBoard) -> Move:
        """
        Choose and return a legal random move.

        Args:
            board: Current HEX board.

        Returns:
            A legal move as a tuple (row, col).

        Raises:
            ValueError: If no legal moves are available.
        """
        legal_moves = self._legal_moves(board.board)

        if not legal_moves:
            raise ValueError("No legal moves available.")

        return self.rng.choice(legal_moves)

    def _legal_moves(self, mat: list[list[int]]) -> list[Move]:
        """
        Collect all empty cells from the board.

        Args:
            mat: Board matrix where 0 means empty.

        Returns:
            A list of legal moves (row, col).
        """
        moves: list[Move] = []
        n = len(mat)

        for r in range(n):
            for c in range(n):
                if mat[r][c] == 0:
                    moves.append((r, c))

        return moves