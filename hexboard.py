from __future__ import annotations

from collections import deque


class HexBoard:
    def __init__(self, size: int):
        self.size = size  # Tamaño N del tablero (NxN)
        self.board = [[0 for _ in range(size)] for _ in range(size)]  # Matriz NxN (0=vacío, 1=Jugador1, 2=Jugador2)

    def clone(self) -> HexBoard:
        """Devuelve una copia del tablero actual"""
        cloned = HexBoard(self.size)
        cloned.board = [row[:] for row in self.board]
        return cloned

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        """Coloca una ficha si la casilla está vacía."""
        if player_id not in (1, 2):
            return False
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.board[row][col] != 0:
            return False

        self.board[row][col] = player_id
        return True

    def check_connection(self, player_id: int) -> bool:
        """Verifica si el jugador ha conectado sus dos lados"""
        if player_id not in (1, 2):
            return False
        if self.size == 0:
            return False

        if player_id == 1:
            starts = [(r, 0) for r in range(self.size) if self.board[r][0] == 1]
            reached_goal = lambda r, c: c == self.size - 1
        else:
            starts = [(0, c) for c in range(self.size) if self.board[0][c] == 2]
            reached_goal = lambda r, c: r == self.size - 1

        queue = deque(starts)
        visited = set(starts)

        while queue:
            row, col = queue.popleft()
            if reached_goal(row, col):
                return True

            for nr, nc in self._neighbors(row, col):
                if (nr, nc) in visited:
                    continue
                if self.board[nr][nc] != player_id:
                    continue
                visited.add((nr, nc))
                queue.append((nr, nc))

        return False

    def _neighbors(self, row: int, col: int):
        if row % 2 == 0:
            deltas = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        else:
            deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                yield nr, nc
