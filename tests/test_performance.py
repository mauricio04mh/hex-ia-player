from __future__ import annotations

import os
import random
import time

from board import HexBoard
from tests.player_under_test import PlayerUnderTest
from tests.utils.hex_utils import is_legal_move, legal_moves
from tests.utils.positions import (
    must_block_player2_wins_in_one,
    winning_in_one_for_player1,
)


def _random_board(size: int, seed: int, fill: float) -> HexBoard:
    rng = random.Random(seed)
    board = HexBoard(size)
    for row in range(size):
        for col in range(size):
            x = rng.random()
            if x < fill / 2:
                board.board[row][col] = 1
            elif x < fill:
                board.board[row][col] = 2
    return board


def _build_benchmark_boards() -> list[HexBoard]:
    tactical1, _ = winning_in_one_for_player1()
    tactical2, _ = must_block_player2_wins_in_one()

    return [
        HexBoard(5),
        HexBoard(7),
        _random_board(5, seed=202, fill=0.35),
        _random_board(5, seed=203, fill=0.55),
        _random_board(7, seed=204, fill=0.35),
        _random_board(7, seed=205, fill=0.50),
        tactical1,
        tactical2,
    ]


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = int(0.95 * (len(ordered) - 1))
    return ordered[idx]


def test_play_performance_budget() -> None:
    max_play_ms = float(os.getenv("HEX_MAX_PLAY_MS", "300"))
    player = PlayerUnderTest(1)
    times_ms: list[float] = []

    boards = _build_benchmark_boards()
    repeats = 3

    for _ in range(repeats):
        for board in boards:
            if not legal_moves(board):
                continue
            start = time.perf_counter()
            move = player.play(board.clone())
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            times_ms.append(elapsed_ms)

            assert is_legal_move(board, move), (
                f"Movimiento ilegal durante benchmark: {move}"
            )

    avg_ms = sum(times_ms) / len(times_ms)
    max_ms = max(times_ms)
    p95_ms = _p95(times_ms)

    assert p95_ms <= max_play_ms, (
        f"Rendimiento fuera de presupuesto. avg={avg_ms:.2f}ms, "
        f"p95={p95_ms:.2f}ms, max={max_ms:.2f}ms, limit={max_play_ms:.2f}ms"
    )
    assert max_ms <= max_play_ms * 1.5, (
        f"Peor caso demasiado alto. avg={avg_ms:.2f}ms, "
        f"p95={p95_ms:.2f}ms, max={max_ms:.2f}ms, limit={max_play_ms:.2f}ms"
    )


def test_play_never_exceeds_hard_timeout() -> None:
    hard_limit_ms = float(os.getenv("HEX_HARD_MAX_PLAY_MS", "4500"))
    player = PlayerUnderTest(1)
    boards = _build_benchmark_boards()
    repeats = 3

    for rep in range(repeats):
        for board_idx, board in enumerate(boards):
            if not legal_moves(board):
                continue

            start = time.perf_counter()
            move = player.play(board.clone())
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            assert is_legal_move(board, move), (
                f"Movimiento ilegal durante hard-timeout check: {move}"
            )
            assert elapsed_ms <= hard_limit_ms, (
                "Timeout duro excedido en una jugada. "
                f"rep={rep}, board_idx={board_idx}, "
                f"elapsed={elapsed_ms:.2f}ms, limit={hard_limit_ms:.2f}ms"
            )
