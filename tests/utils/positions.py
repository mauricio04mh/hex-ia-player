from __future__ import annotations

from board import HexBoard

from tests.utils.hex_utils import (
    board_from_matrix,
    immediate_winning_moves,
    legal_moves,
    move_allows_opponent_win_in_one,
    opponent_of,
    simulate_move_and_check_win,
)

Move = tuple[int, int]


def winning_in_one_for_player1() -> tuple[HexBoard, set[Move]]:
    """Jugador 1 tiene una jugada ganadora inmediata."""
    matrix = [
        [0, 0, 2],
        [1, 1, 0],
        [0, 0, 2],
    ]
    board = board_from_matrix(matrix)

    wins = immediate_winning_moves(board, 1)
    assert wins, "La posición debe tener al menos una victoria inmediata para J1"
    assert not board.check_connection(1)
    return board, wins


def must_block_player2_wins_in_one() -> tuple[HexBoard, set[Move]]:
    """Jugador 2 amenaza con ganar en 1; J1 debe bloquear."""
    matrix = [
        [0, 2, 0],
        [0, 2, 0],
        [0, 0, 1],
    ]
    board = board_from_matrix(matrix)

    opp_wins = immediate_winning_moves(board, 2)
    assert opp_wins, "La posición debe tener amenaza inmediata de J2"

    blocking_moves: set[Move] = set()
    for move in legal_moves(board):
        cloned = board.clone()
        cloned.place_piece(move[0], move[1], 1)
        if not immediate_winning_moves(cloned, 2):
            blocking_moves.add(move)

    assert blocking_moves, "Debe existir al menos un bloqueo para J1"
    return board, blocking_moves


def prioritize_win_over_block_for_player1() -> tuple[HexBoard, set[Move], set[Move]]:
    """
    J1 y J2 tienen victoria en 1.
    El test exige que J1 elija jugada ganadora para sí mismo.
    """
    matrix = [
        [0, 1, 0, 1, 0],
        [2, 1, 0, 2, 0],
        [0, 2, 1, 1, 1],
        [0, 2, 1, 2, 0],
        [0, 2, 0, 2, 1],
    ]
    board = board_from_matrix(matrix)

    p1_wins = immediate_winning_moves(board, 1)
    p2_wins = immediate_winning_moves(board, 2)

    assert p1_wins, "La posición debe tener victoria inmediata para J1"
    assert p2_wins, "La posición debe tener amenaza inmediata de J2"
    return board, p1_wins, p2_wins


def avoid_giving_immediate_loss_for_player1() -> tuple[HexBoard, set[Move], set[Move]]:
    """
    Existen jugadas de J1 que dejan victoria inmediata a J2, y otras seguras.
    """
    matrix = [
        [0, 0, 0, 1, 2],
        [2, 0, 1, 1, 2],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 1, 2],
        [0, 0, 0, 0, 0],
    ]
    board = board_from_matrix(matrix)

    dangerous: set[Move] = set()
    safe: set[Move] = set()
    for move in legal_moves(board):
        if move_allows_opponent_win_in_one(board, move, 1):
            dangerous.add(move)
        else:
            safe.add(move)

    assert dangerous, "La posición debe tener jugadas peligrosas"
    assert safe, "La posición debe tener alternativas seguras"
    return board, dangerous, safe


def regression_cases() -> list[dict]:
    """Casos estables para regresión táctica del bot."""
    board1, wins1 = winning_in_one_for_player1()
    board2, blocks2 = must_block_player2_wins_in_one()
    board3, wins3, _ = prioritize_win_over_block_for_player1()

    return [
        {
            "name": "win-in-one-j1",
            "player_id": 1,
            "board": board1,
            "expected_moves": wins1,
            "description": "Debe tomar victoria inmediata",
        },
        {
            "name": "must-block-j2-threat",
            "player_id": 1,
            "board": board2,
            "expected_moves": blocks2,
            "description": "Debe bloquear amenaza inmediata",
        },
        {
            "name": "prioritize-win-when-threatened",
            "player_id": 1,
            "board": board3,
            "expected_moves": wins3,
            "description": "Debe priorizar ganar ahora",
        },
    ]


def verify_move_is_winning(board: HexBoard, move: Move, player_id: int) -> bool:
    return simulate_move_and_check_win(board, move, player_id)


def verify_move_blocks_immediate_win(board: HexBoard, move: Move, player_id: int) -> bool:
    cloned = board.clone()
    cloned.place_piece(move[0], move[1], player_id)
    opponent_id = opponent_of(player_id)
    return not immediate_winning_moves(cloned, opponent_id)
