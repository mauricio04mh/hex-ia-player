"""Punto unico para elegir que SmartPlayer se prueba en toda la suite.

Cambia SOLO este import para apuntar a otra version de tu bot.
"""

from players.player_3 import SmartPlayer as PlayerUnderTest

__all__ = ["PlayerUnderTest"]
