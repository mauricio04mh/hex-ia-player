from __future__ import annotations

from tests.player_under_test import PlayerUnderTest
from tests.utils.hex_utils import is_legal_move
from tests.utils.positions import (
    avoid_giving_immediate_loss_for_player1,
    must_block_player2_wins_in_one,
    prioritize_win_over_block_for_player1,
    verify_move_blocks_immediate_win,
    verify_move_is_winning,
    winning_in_one_for_player1,
)


def test_tactic_win_in_one() -> None:
    board, expected_winning_moves = winning_in_one_for_player1()
    move = PlayerUnderTest(1).play(board)

    assert is_legal_move(board, move)
    assert move in expected_winning_moves, (
        f"Debía jugar una ganadora inmediata. Esperadas={sorted(expected_winning_moves)}, obtuvo={move}"
    )
    assert verify_move_is_winning(board, move, 1)


def test_tactic_block_opponent_win_in_one() -> None:
    board, expected_block_moves = must_block_player2_wins_in_one()
    move = PlayerUnderTest(1).play(board)

    assert is_legal_move(board, move)
    assert move in expected_block_moves, (
        f"Debía bloquear. Bloqueos válidos={sorted(expected_block_moves)}, obtuvo={move}"
    )
    assert verify_move_blocks_immediate_win(board, move, 1)


def test_tactic_prioritize_winning_move() -> None:
    board, winning_moves, opponent_winning_moves = prioritize_win_over_block_for_player1()
    move = PlayerUnderTest(1).play(board)

    assert opponent_winning_moves, "El rival debe tener amenaza inmediata en este caso"
    assert is_legal_move(board, move)
    assert move in winning_moves, (
        f"Si puede ganar en 1 debe priorizar ganar. Ganadoras={sorted(winning_moves)}, obtuvo={move}"
    )


def test_tactic_avoid_giving_immediate_loss_when_safe_exists() -> None:
    board, dangerous_moves, safe_moves = avoid_giving_immediate_loss_for_player1()
    move = PlayerUnderTest(1).play(board)

    assert is_legal_move(board, move)
    assert safe_moves, "Debe existir al menos una jugada segura"
    assert move in safe_moves, (
        f"No debe regalar victoria inmediata si hay alternativa segura. "
        f"Peligrosas={sorted(dangerous_moves)}, seguras={sorted(safe_moves)}, obtuvo={move}"
    )
