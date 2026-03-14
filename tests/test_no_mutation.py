from __future__ import annotations

import random

import pytest

from board import HexBoard
from tests.player_under_test import PlayerUnderTest
from tests.utils.hex_utils import deepcopy_board
from tests.utils.positions import (
    must_block_player2_wins_in_one,
    winning_in_one_for_player1,
)


def _random_board(size: int, seed: int) -> HexBoard:
    rng = random.Random(seed)
    board = HexBoard(size)
    for row in range(size):
        for col in range(size):
            x = rng.random()
            if x < 0.30:
                board.board[row][col] = 1
            elif x < 0.60:
                board.board[row][col] = 2
    return board


@pytest.mark.parametrize("size", [3, 5, 7])
@pytest.mark.parametrize("player_id", [1, 2])
def test_play_does_not_mutate_input_board_random(size: int, player_id: int) -> None:
    board = _random_board(size, seed=9000 + size * 10 + player_id)
    before = deepcopy_board(board)

    PlayerUnderTest(player_id).play(board)

    assert board.board == before, "play() no debe mutar el HexBoard recibido"


@pytest.mark.parametrize("player_id", [1, 2])
def test_play_does_not_mutate_input_board_tactical(player_id: int) -> None:
    board1, _ = winning_in_one_for_player1()
    board2, _ = must_block_player2_wins_in_one()

    for board in (board1, board2):
        before = deepcopy_board(board)
        PlayerUnderTest(player_id).play(board)
        assert board.board == before, "play() mutó el tablero táctico de entrada"
