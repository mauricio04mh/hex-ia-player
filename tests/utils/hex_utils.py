from __future__ import annotations

from typing import Sequence

from board import HexBoard

Move = tuple[int, int]


def deepcopy_board(board: HexBoard) -> list[list[int]]:
    """Snapshot profundo de la matriz interna del tablero."""
    return [row[:] for row in board.board]


def legal_moves(board: HexBoard) -> list[Move]:
    return [
        (row, col)
        for row in range(board.size)
        for col in range(board.size)
        if board.board[row][col] == 0
    ]


def is_legal_move(board: HexBoard, move: object) -> bool:
    if not isinstance(move, tuple) or len(move) != 2:
        return False

    row, col = move
    if not isinstance(row, int) or not isinstance(col, int):
        return False
    if not (0 <= row < board.size and 0 <= col < board.size):
        return False
    return board.board[row][col] == 0


def apply_move(board: HexBoard, move: Move, player_id: int) -> bool:
    row, col = move
    return board.place_piece(row, col, player_id)


def simulate_move_and_check_win(board: HexBoard, move: Move, player_id: int) -> bool:
    cloned = board.clone()
    if not apply_move(cloned, move, player_id):
        return False
    return cloned.check_connection(player_id)


def board_from_matrix(matrix: Sequence[Sequence[int]]) -> HexBoard:
    size = len(matrix)
    if size == 0:
        raise ValueError("matrix must be non-empty")

    for row in matrix:
        if len(row) != size:
            raise ValueError("matrix must be square (NxN)")
        for value in row:
            if value not in (0, 1, 2):
                raise ValueError("matrix values must be 0, 1 or 2")

    board = HexBoard(size)
    board.board = [list(row) for row in matrix]
    return board


def immediate_winning_moves(board: HexBoard, player_id: int) -> set[Move]:
    return {
        move
        for move in legal_moves(board)
        if simulate_move_and_check_win(board, move, player_id)
    }


def opponent_of(player_id: int) -> int:
    if player_id not in (1, 2):
        raise ValueError("player_id must be 1 or 2")
    return 2 if player_id == 1 else 1


def board_is_full(board: HexBoard) -> bool:
    return all(cell != 0 for row in board.board for cell in row)


def move_allows_opponent_win_in_one(board: HexBoard, move: Move, player_id: int) -> bool:
    cloned = board.clone()
    if not apply_move(cloned, move, player_id):
        return True
    opp = opponent_of(player_id)
    return bool(immediate_winning_moves(cloned, opp))
