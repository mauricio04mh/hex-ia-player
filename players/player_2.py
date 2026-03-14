from __future__ import annotations

"""
solution.py

SmartPlayer V2 - Iteration 2 for HEX
====================================

This player implements the second serious iteration of a HEX agent:

1) Tactical layer from Iteration 1:
   - If we can win immediately, do it.
   - If the opponent can win immediately on their next move, block it.

2) Strategic layer from Iteration 2:
   - If there is no forced tactical move, evaluate each legal move with
     a 1-ply lookahead using shortest-path evaluation.
   - The board is modeled as a graph on the hex cells.
   - We run multi-source Dijkstra from one target border to the opposite one.

Cell costs for a player P:
   - P's own stone    -> cost 0
   - empty cell       -> cost 1
   - opponent stone   -> INF (blocked)

This means the shortest-path distance estimates:
   "How many empty cells does this player still need, approximately,
    to connect their two target sides?"

Evaluation function:
   score = opp_dist - my_dist

So:
   - smaller my_dist is better
   - larger opp_dist is better
   - larger score is better for us

Why this is stronger than Iteration 1:
   - It gives global board awareness.
   - It values connection plans, not just local shape.
   - It is still much cheaper than deep minimax or MCTS.

Important:
   - This is still not deep search.
   - It is a 1-ply evaluator: try move -> evaluate resulting position.
   - Immediate tactics are still handled first before the Dijkstra layer.
"""

import heapq

from board import HexBoard
from player import Player


Move = tuple[int, int]


class SmartPlayer(Player):
    """
    HEX player for Iteration 2:
    tactical priority first, then 1-ply Dijkstra evaluation.
    """

    # Very large cost used to represent "blocked by opponent".
    INF = 10**9

    def __init__(self, player_id: int):
        """
        Initialize the player.

        Args:
            player_id:
                1 -> connects left to right
                2 -> connects top to bottom
        """
        super().__init__(player_id)
        self.opponent_id = 2 if player_id == 1 else 1

    def play(self, board: HexBoard) -> Move:
        """
        Choose a move on the current board.

        Priority:
        1) Win now if possible.
        2) Block opponent's immediate win if needed.
        3) Otherwise, choose the move with the best 1-ply Dijkstra score.

        Args:
            board: Current HEX board.

        Returns:
            A legal move (row, col).
        """
        legal_moves = self._legal_moves(board.board)
        if not legal_moves:
            raise ValueError("No legal moves available.")

        # ------------------------------------------------------------
        # 1) Immediate tactical win.
        # ------------------------------------------------------------
        for move in legal_moves:
            if self._wins_after_move(board, move, self.player_id):
                return move

        # ------------------------------------------------------------
        # 2) Immediate tactical defense.
        #    If opponent could win by playing some cell next turn,
        #    we must occupy that cell now.
        # ------------------------------------------------------------
        forced_blocks = [
            move
            for move in legal_moves
            if self._wins_after_move(board, move, self.opponent_id)
        ]

        if forced_blocks:
            return self._best_move_by_dijkstra(board, forced_blocks)

        # ------------------------------------------------------------
        # 3) No immediate tactics -> strategic evaluation with
        #    1-ply shortest-path score.
        # ------------------------------------------------------------
        return self._best_move_by_dijkstra(board, legal_moves)

    def _legal_moves(self, mat: list[list[int]]) -> list[Move]:
        """
        Return all legal moves (all empty cells).

        Args:
            mat: NxN board matrix.

        Returns:
            List of (row, col) for empty cells.
        """
        moves: list[Move] = []
        n = len(mat)

        for r in range(n):
            for c in range(n):
                if mat[r][c] == 0:
                    moves.append((r, c))

        return moves

    def _wins_after_move(self, board: HexBoard, move: Move, pid: int) -> bool:
        """
        Check whether player `pid` wins immediately by playing `move`.

        Args:
            board: Current board.
            move: Candidate move (row, col).
            pid: Player to test.

        Returns:
            True if that move wins immediately, otherwise False.
        """
        sim = board.clone()

        if not sim.place_piece(move[0], move[1], pid):
            return False

        return sim.check_connection(pid)

    def _best_move_by_dijkstra(self, board: HexBoard, moves: list[Move]) -> Move:
        """
        Among the given candidate moves, choose the one with the best
        shortest-path evaluation after simulating our move.

        We evaluate each move with:
            score = opp_dist - my_dist

        Tie-break:
            1) better score
            2) more central move
            3) lexicographically smaller coordinate for determinism

        Args:
            board: Current board.
            moves: Candidate legal moves.

        Returns:
            Best move according to 1-ply Dijkstra evaluation.
        """
        best_move = moves[0]
        best_score = -self.INF
        best_center_penalty = self._center_penalty(board.board, best_move)

        for move in moves:
            sim = board.clone()
            ok = sim.place_piece(move[0], move[1], self.player_id)
            if not ok:
                continue

            score = self._evaluate_position(sim.board)
            center_penalty = self._center_penalty(board.board, move)

            if score > best_score:
                best_score = score
                best_move = move
                best_center_penalty = center_penalty
            elif score == best_score:
                # Prefer more central moves when the Dijkstra score ties.
                if center_penalty < best_center_penalty:
                    best_move = move
                    best_center_penalty = center_penalty
                elif center_penalty == best_center_penalty and move < best_move:
                    best_move = move

        return best_move

    def _evaluate_position(self, mat: list[list[int]]) -> int:
        """
        Evaluate the board position from our perspective.

        The evaluation is:
            score = opp_dist - my_dist

        Larger is better for us.

        Args:
            mat: Board matrix.

        Returns:
            Integer evaluation score.
        """
        my_dist = self._shortest_connection_distance(mat, self.player_id)
        opp_dist = self._shortest_connection_distance(mat, self.opponent_id)

        return opp_dist - my_dist

    def _shortest_connection_distance(self, mat: list[list[int]], pid: int) -> int:
        """
        Compute the shortest connection distance for player `pid`
        using multi-source Dijkstra.

        Interpretation:
            - own stones cost 0
            - empty cells cost 1
            - opponent stones are blocked (INF)

        This estimates how many empty cells are still needed, approximately,
        for `pid` to connect their two target borders.

        For player 1:
            start border = left column
            goal border  = right column

        For player 2:
            start border = top row
            goal border  = bottom row

        Args:
            mat: Board matrix.
            pid: Player ID whose distance is being computed.

        Returns:
            Minimum connection cost, or INF if effectively blocked.
        """
        n = len(mat)
        dist = [[self.INF] * n for _ in range(n)]
        pq: list[tuple[int, int, int]] = []

        # ------------------------------------------------------------
        # Initialize all valid start-border cells as Dijkstra sources.
        # This is "multi-source" Dijkstra.
        # ------------------------------------------------------------
        for r, c in self._start_border_cells(n, pid):
            start_cost = self._cell_cost(mat[r][c], pid)
            if start_cost >= self.INF:
                continue

            if start_cost < dist[r][c]:
                dist[r][c] = start_cost
                heapq.heappush(pq, (start_cost, r, c))

        # If no border entry is usable, return INF.
        if not pq:
            return self.INF

        # ------------------------------------------------------------
        # Standard Dijkstra over the hex-cell graph.
        # ------------------------------------------------------------
        while pq:
            cur_dist, r, c = heapq.heappop(pq)

            if cur_dist != dist[r][c]:
                continue

            # First time we pop a goal-border cell, we have the minimum cost.
            if self._is_goal_border_cell(r, c, n, pid):
                return cur_dist

            for nr, nc in self._neighbors_evenr(n, r, c):
                step_cost = self._cell_cost(mat[nr][nc], pid)
                if step_cost >= self.INF:
                    continue

                new_dist = cur_dist + step_cost
                if new_dist < dist[nr][nc]:
                    dist[nr][nc] = new_dist
                    heapq.heappush(pq, (new_dist, nr, nc))

        return self.INF

    def _cell_cost(self, cell_value: int, pid: int) -> int:
        """
        Return the traversal cost of a cell from player `pid`'s perspective.

        Rules:
            own stone   -> 0
            empty cell  -> 1
            opponent    -> INF

        Args:
            cell_value: Value stored in the board cell.
            pid: Perspective player.

        Returns:
            Integer traversal cost.
        """
        if cell_value == pid:
            return 0
        if cell_value == 0:
            return 1
        return self.INF

    def _start_border_cells(self, n: int, pid: int) -> list[Move]:
        """
        Return all cells on the start border for player `pid`.

        Player 1: left border
        Player 2: top border

        Args:
            n: Board size.
            pid: Player ID.

        Returns:
            List of border cells.
        """
        if pid == 1:
            return [(r, 0) for r in range(n)]
        return [(0, c) for c in range(n)]

    def _is_goal_border_cell(self, r: int, c: int, n: int, pid: int) -> bool:
        """
        Check whether a cell belongs to the goal border for player `pid`.

        Player 1: right border
        Player 2: bottom border

        Args:
            r: Row index.
            c: Column index.
            n: Board size.
            pid: Player ID.

        Returns:
            True if the cell is on the goal border, else False.
        """
        if pid == 1:
            return c == n - 1
        return r == n - 1

    def _neighbors_evenr(self, n: int, r: int, c: int) -> list[Move]:
        """
        Return the valid hex neighbors of (r, c) using even-r layout.

        Even-r offset coordinates use different neighbor patterns depending
        on whether the row is even or odd.

        Args:
            n: Board size.
            r: Row index.
            c: Column index.

        Returns:
            List of valid neighboring cells.
        """
        if r % 2 == 0:
            deltas = [
                (-1, -1), (-1, 0),
                (0, -1),  (0, 1),
                (1, -1),  (1, 0),
            ]
        else:
            deltas = [
                (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, 0),  (1, 1),
            ]

        out: list[Move] = []
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                out.append((nr, nc))

        return out

    def _center_penalty(self, mat: list[list[int]], move: Move) -> float:
        """
        Small tie-break helper: prefer more central moves when the main
        Dijkstra score is tied.

        Lower penalty is better.

        Args:
            mat: Board matrix.
            move: Candidate move.

        Returns:
            Squared Euclidean distance to board center.
        """
        n = len(mat)
        r, c = move
        center = (n - 1) / 2.0
        return (r - center) ** 2 + (c - center) ** 2