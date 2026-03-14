from __future__ import annotations

import pytest

from tests.player_under_test import PlayerUnderTest
from tests.utils.hex_utils import is_legal_move
from tests.utils.positions import regression_cases


@pytest.mark.parametrize("case", regression_cases(), ids=lambda case: case["name"])
def test_regression_critical_positions(case: dict) -> None:
    player = PlayerUnderTest(case["player_id"])
    board = case["board"].clone()
    move = player.play(board)

    assert is_legal_move(board, move), (
        f"{case['name']}: jugada ilegal ({case['description']}). move={move}"
    )
    assert move in case["expected_moves"], (
        f"{case['name']}: esperaba uno de {sorted(case['expected_moves'])}, obtuvo {move}. "
        f"Contexto: {case['description']}"
    )


def test_regression_cases_have_expected_moves() -> None:
    for case in regression_cases():
        assert case["expected_moves"], f"{case['name']} no tiene movimientos esperados"
