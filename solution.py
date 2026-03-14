from __future__ import annotations

from player import Player
from board import HexBoard

from typing import List, Tuple


Move = Tuple[int, int]

class SmartPlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.opponent_id = 2 if player_id == 1 else 1

    def play(self, board: HexBoard) -> Move:
        pass

    