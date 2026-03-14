from __future__ import annotations

import random

import pytest

from board import HexBoard
from tests.player_under_test import PlayerUnderTest
from tests.utils.hex_utils import is_legal_move, legal_moves


def _midgame_board(size: int, seed: int) -> HexBoard:
    rng = random.Random(seed)
    board = HexBoard(size)
    for row in range(size):
        for col in range(size):
            x = rng.random()
            if x < 0.25:
                board.board[row][col] = 1
            elif x < 0.50:
                board.board[row][col] = 2
    return board


def _almost_full_board(size: int) -> HexBoard:
    board = HexBoard(size)
    value = 1
    for row in range(size):
        for col in range(size):
            board.board[row][col] = value
            value = 2 if value == 1 else 1
    board.board[size // 2][size // 2] = 0
    return board


@pytest.mark.parametrize("size", [3, 5, 7])
@pytest.mark.parametrize("player_id", [1, 2])
def test_play_returns_legal_move_on_representative_states(size: int, player_id: int) -> None:
    player = PlayerUnderTest(player_id)

    boards = [
        HexBoard(size),
        _midgame_board(size, seed=100 + size + player_id),
        _almost_full_board(size),
    ]

    for board in boards:
        if not legal_moves(board):
            continue
        move = player.play(board)
        assert is_legal_move(board, move), (
            f"Movimiento ilegal para size={size}, player_id={player_id}: {move}"
        )


@pytest.mark.parametrize("size", [3, 5, 7])
def test_play_is_robust_on_multiple_random_states(size: int) -> None:
    rng = random.Random(12345 + size)

    for player_id in (1, 2):
        player = PlayerUnderTest(player_id)
        for _ in range(8):
            board = HexBoard(size)
            for row in range(size):
                for col in range(size):
                    x = rng.random()
                    if x < 0.28:
                        board.board[row][col] = 1
                    elif x < 0.56:
                        board.board[row][col] = 2

            if not legal_moves(board):
                continue
            move = player.play(board)
            assert is_legal_move(board, move), (
                f"Movimiento ilegal/robustez fallida en estado aleatorio: {move}"
            )
