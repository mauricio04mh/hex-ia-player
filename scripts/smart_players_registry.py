"""Registro de versiones SmartPlayer para la arena.

Agrega nuevas versiones importandolas aqui y anadiendolas a SMART_PLAYERS.
La arena exige al menos 2 entradas.
"""

from solution import SmartPlayer as SmartPlayerV1
from v0 import SmartPlayer as VopLAYER
from v0 import SmartPlayer as player_aux
from minimax_v0 import SmartPlayer as MinimaxV0Player
from minimax_v1 import SmartPlayer as MinimaxV1Player
from minimax_v2 import SmartPlayer as MinimaxV2Player
from minimax_v3_guided import SmartPlayer as MinimaxV3Player
from players.player_1 import SmartPlayer as Player1
from players.player_0 import RandomPlayer
from players.player_2 import SmartPlayer as Player2
from players.player_3 import SmartPlayer as Player3
from players.player_4 import SmartPlayer as Player4
from players.player_4_improved import SmartPlayer as Player4Improved


SMART_PLAYERS: list[tuple[str, type]] = [
    # ("v0", VopLAYER),
    # ("minimax_v0", MinimaxV0Player),
    # ("minimax_v1", MinimaxV1Player),
    # ("minimax_v2", MinimaxV2Player),
    # ("minimax_v3", MinimaxV3Player),
    # ("player_1", Player1),
    # ("random_player", RandomPlayer),
    ("player_2", Player2),
    ("player_4", Player4),
    ("player_4.1", Player4Improved),
    # ("v3", SmartPlayerV3),
]
