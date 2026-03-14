from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from board import HexBoard
from scripts.smart_players_registry import SMART_PLAYERS
from tests.utils.hex_utils import board_is_full, is_legal_move, opponent_of


@dataclass
class PlayerStats:
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    forfeits: int = 0
    move_times_ms: list[float] = field(default_factory=list)

    @property
    def points(self) -> int:
        return self.wins

    @property
    def winrate(self) -> float:
        return (self.wins / self.games * 100.0) if self.games else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(q * (len(ordered) - 1))
    return ordered[idx]


def _time_stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    avg = sum(values) / len(values)
    p95 = _percentile(values, 0.95)
    mx = max(values)
    return avg, p95, mx


def _validate_registry() -> list[tuple[str, type]]:
    if len(SMART_PLAYERS) < 2:
        raise RuntimeError(
            "Debes registrar al menos 2 SmartPlayers en scripts/smart_players_registry.py"
        )

    names = [name for name, _ in SMART_PLAYERS]
    if len(set(names)) != len(names):
        raise RuntimeError("Los nombres en SMART_PLAYERS deben ser unicos")

    for name, cls in SMART_PLAYERS:
        if not callable(cls):
            raise RuntimeError(f"La entrada '{name}' no es una clase/callable valida")

    return SMART_PLAYERS


def _parse_sizes(sizes_raw: str) -> list[int]:
    sizes: list[int] = []
    for chunk in sizes_raw.split(","):
        value = int(chunk.strip())
        if value < 2:
            raise ValueError("Cada size debe ser >= 2")
        sizes.append(value)

    if not sizes:
        raise ValueError("Debes indicar al menos un tamano en --sizes")
    return sizes


def _run_single_game(
    size: int,
    p1_name: str,
    p1_cls: type,
    p2_name: str,
    p2_cls: type,
) -> tuple[str, dict[str, list[float]], str | None]:
    board = HexBoard(size)
    players = {1: p1_cls(1), 2: p2_cls(2)}
    names = {1: p1_name, 2: p2_name}
    timings = {p1_name: [], p2_name: []}

    turn = 1
    while not board_is_full(board):
        view = board.clone()
        start = time.perf_counter()
        move = players[turn].play(view)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings[names[turn]].append(elapsed_ms)

        if not is_legal_move(board, move):
            winner = opponent_of(turn)
            return names[winner], timings, names[turn]

        board.place_piece(move[0], move[1], turn)
        if board.check_connection(turn):
            return names[turn], timings, None

        turn = opponent_of(turn)

    return "draw", timings, None


def run_tournament(sizes: list[int], games_per_pair: int, seed: int) -> dict[str, PlayerStats]:
    registry = _validate_registry()
    stats: dict[str, PlayerStats] = {name: PlayerStats() for name, _ in registry}
    rng = random.Random(seed)

    total_pairs = len(list(combinations(registry, 2)))
    total_games = total_pairs * len(sizes) * games_per_pair
    print(
        f"Torneo Smart vs Smart | players={len(registry)} pairs={total_pairs} "
        f"sizes={sizes} games_per_pair={games_per_pair} total_games={total_games}"
    )

    for size in sizes:
        print(f"\n[SIZE {size}]")
        for (name_a, cls_a), (name_b, cls_b) in combinations(registry, 2):
            for game_idx in range(games_per_pair):
                random.seed(rng.randint(0, 2**31 - 1))

                if game_idx % 2 == 0:
                    p1_name, p1_cls = name_a, cls_a
                    p2_name, p2_cls = name_b, cls_b
                else:
                    p1_name, p1_cls = name_b, cls_b
                    p2_name, p2_cls = name_a, cls_a

                winner, timings, forfeit = _run_single_game(
                    size=size,
                    p1_name=p1_name,
                    p1_cls=p1_cls,
                    p2_name=p2_name,
                    p2_cls=p2_cls,
                )

                stats[name_a].games += 1
                stats[name_b].games += 1
                stats[name_a].move_times_ms.extend(timings[name_a])
                stats[name_b].move_times_ms.extend(timings[name_b])

                if winner == "draw":
                    stats[name_a].draws += 1
                    stats[name_b].draws += 1
                elif winner == name_a:
                    stats[name_a].wins += 1
                    stats[name_b].losses += 1
                elif winner == name_b:
                    stats[name_b].wins += 1
                    stats[name_a].losses += 1
                else:
                    raise RuntimeError(f"Ganador inesperado: {winner}")

                if forfeit is not None:
                    stats[forfeit].forfeits += 1

    return stats


def print_ranking(stats: dict[str, PlayerStats]) -> None:
    ordered = sorted(
        stats.items(),
        key=lambda item: (
            item[1].points,
            item[1].wins,
            item[1].winrate,
            -item[1].forfeits,
        ),
        reverse=True,
    )

    print("\n=== CLASIFICACION ===")
    print(
        "pos  player        pts  games  wins  draws  losses  forfeits  winrate   "
        "t_avg(ms)  t_p95(ms)  t_max(ms)"
    )

    for idx, (name, st) in enumerate(ordered, start=1):
        t_avg, t_p95, t_max = _time_stats(st.move_times_ms)
        print(
            f"{idx:>3}  {name:<12}  {st.points:>3}  {st.games:>5}  {st.wins:>4}  "
            f"{st.draws:>5}  {st.losses:>6}  {st.forfeits:>8}  {st.winrate:>7.2f}%  "
            f"{t_avg:>9.2f}  {t_p95:>9.2f}  {t_max:>9.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Liga local entre multiples SmartPlayers registrados"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="5,7",
        help="Lista de tamanos separada por coma, ej: 3,5,7",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Partidas por emparejamiento y por tamano",
    )
    parser.add_argument("--seed", type=int, default=0, help="Semilla reproducible")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = _parse_sizes(args.sizes)
    stats = run_tournament(sizes=sizes, games_per_pair=args.games, seed=args.seed)
    print_ranking(stats)


if __name__ == "__main__":
    main()
