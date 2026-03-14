from __future__ import annotations

"""
solution.py

SmartPlayer V3 - Iteration 3 for HEX
====================================

This player implements the third iteration of the project roadmap:

1) Tactical priority:
   - If we can win immediately, do it.
   - If the opponent can win immediately next turn, block it.

2) Candidate generation:
   - Score all legal moves with a CHEAP heuristic.
   - Sort them from most promising to least promising.
   - Keep only the Top-K moves.

3) Strategic selection:
   - Evaluate only those Top-K candidates with the stronger
     shortest-path / Dijkstra evaluation from Iteration 2.
   - Choose the best candidate.

Why this is Iteration 3 and not just Iteration 2:
- Iteration 2 evaluated every legal move with the expensive evaluator.
- Iteration 3 first prunes the move list with a quick scorer, then
  uses the expensive evaluator only on a small shortlist.

This reduces time per move and prepares the code structure for:
- alpha-beta,
- iterative deepening,
- better move ordering later.

Main idea of the Dijkstra evaluation:
- Our stones cost 0
- Empty cells cost 1
- Opponent stones cost INF

The shortest-path distance estimates how many empty cells are still
needed, approximately, to connect the two target borders.

Evaluation:
    score = opp_dist - my_dist

Larger score is better for us.
"""

import heapq

from board import HexBoard
from player import Player


Move = tuple[int, int]


class SmartPlayer(Player):
    """
    HEX player for Iteration 3:
    tactical rules + Top-K pruning + Dijkstra selection.
    """

    # Large constant used to represent blocked cells in Dijkstra.
    INF = 10**9

    # Default number of candidate moves kept after quick ranking.
    DEFAULT_TOP_K = 10

    # Weights for the cheap move-ranking heuristic.
    QUICK_CENTER_WEIGHT = -1.0
    QUICK_OWN_NEIGHBOR_WEIGHT = 2.5
    QUICK_OPP_NEIGHBOR_WEIGHT = 0.8
    QUICK_AXIS_PROGRESS_WEIGHT = 3.0
    QUICK_GOAL_BORDER_BONUS = 2.0

    def __init__(self, player_id: int, top_k: int | None = None):
        """
        Initialize the player.

        Args:
            player_id:
                1 -> connects left to right
                2 -> connects top to bottom
            top_k:
                Number of moves to keep after quick ranking.
                If None, DEFAULT_TOP_K is used.
        """
        super().__init__(player_id)
        self.opponent_id = 2 if player_id == 1 else 1
        self.top_k = top_k if top_k is not None else self.DEFAULT_TOP_K

    def play(self, board: HexBoard) -> Move:
        """
        Choose a move on the current board.

        Decision order:
        1) Win immediately if possible.
        2) Block opponent's immediate win if necessary.
        3) Rank all legal moves with a cheap heuristic.
        4) Keep only Top-K moves.
        5) Evaluate only those candidates with Dijkstra.
        6) Return the best one.

        Args:
            board: Current HEX board.

        Returns:
            A legal move (row, col).
        """
        legal_moves = self._legal_moves(board.board)
        if not legal_moves:
            raise ValueError("No legal moves available.")

        # ------------------------------------------------------------
        # 1) Tactical win-now.
        # ------------------------------------------------------------
        for move in legal_moves:
            if self._wins_after_move(board, move, self.player_id):
                return move

        # ------------------------------------------------------------
        # 2) Tactical block-now.
        #    These moves must never be pruned away.
        # ------------------------------------------------------------
        forced_blocks = [
            move
            for move in legal_moves
            if self._wins_after_move(board, move, self.opponent_id)
        ]
        if forced_blocks:
            return self._best_move_by_dijkstra(board, forced_blocks)

        # ------------------------------------------------------------
        # 3) Generate Top-K candidates using a cheap ranking function.
        # ------------------------------------------------------------
        candidates = self._top_k_candidates(board.board, legal_moves)

        # ------------------------------------------------------------
        # 4) Among those Top-K candidates, choose the best one using
        #    the more expensive Dijkstra evaluation.
        # ------------------------------------------------------------
        return self._best_move_by_dijkstra(board, candidates)

    def _legal_moves(self, mat: list[list[int]]) -> list[Move]:
        """
        Return all empty cells on the board.

        Args:
            mat: Board matrix.

        Returns:
            List of legal moves (row, col).
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
            move: Candidate move.
            pid: Player to test.

        Returns:
            True if the move wins immediately, otherwise False.
        """
        sim = board.clone()

        if not sim.place_piece(move[0], move[1], pid):
            return False

        return sim.check_connection(pid)

    def _top_k_candidates(self, mat: list[list[int]], legal_moves: list[Move]) -> list[Move]:
        """
        Rank all legal moves with the cheap heuristic and keep only Top-K.

        This is the main Iteration 3 idea:
        - all legal moves are scored cheaply,
        - only the most promising K moves survive,
        - expensive evaluation is reserved for those.

        Args:
            mat: Current board matrix.
            legal_moves: All legal moves.

        Returns:
            The Top-K candidate moves, sorted from best quick-score to worst.
        """
        axis_min, axis_max = self._current_goal_axis_bounds(mat)

        ranked = sorted(
            legal_moves,
            key=lambda m: (
                -self._quick_score_move(mat, m, axis_min, axis_max),
                m[0],
                m[1],
            ),
        )

        k = min(self.top_k, len(ranked))
        return ranked[:k]

    def _quick_score_move(
        self,
        mat: list[list[int]],
        move: Move,
        axis_min: int | None,
        axis_max: int | None,
    ) -> float:
        """
        Cheap heuristic used only for candidate ranking.

        Features:
        A) Centrality:
           More central moves are usually more flexible.

        B) Local connectivity:
           Reward moves adjacent to our stones.
           Give a smaller bonus for adjacency to opponent stones because
           contested zones are important.

        C) Goal-axis progress:
           Reward moves that expand our current span along the axis we need:
           - player 1 wants to expand across columns
           - player 2 wants to expand across rows

        D) Goal-border contact:
           Small bonus for directly touching a target border.

        This heuristic is intentionally cheaper than Dijkstra because its
        job is only to generate a good shortlist, not to be the final judge.

        Args:
            mat: Current board matrix.
            move: Candidate move.
            axis_min: Current minimum occupied coordinate on our goal axis.
            axis_max: Current maximum occupied coordinate on our goal axis.

        Returns:
            Numeric quick score. Higher is better.
        """
        n = len(mat)
        r, c = move
        score = 0.0

        # A) Centrality
        center = (n - 1) / 2.0
        dist2 = (r - center) ** 2 + (c - center) ** 2
        score += self.QUICK_CENTER_WEIGHT * dist2

        # B) Local connectivity / contest
        own_neighbors = 0
        opp_neighbors = 0

        for nr, nc in self._neighbors_evenr(n, r, c):
            cell = mat[nr][nc]
            if cell == self.player_id:
                own_neighbors += 1
            elif cell == self.opponent_id:
                opp_neighbors += 1

        score += self.QUICK_OWN_NEIGHBOR_WEIGHT * own_neighbors
        score += self.QUICK_OPP_NEIGHBOR_WEIGHT * opp_neighbors

        # C) Goal-axis progress
        axis_value = c if self.player_id == 1 else r

        if axis_min is not None and axis_max is not None:
            current_span = axis_max - axis_min
            new_span = max(axis_max, axis_value) - min(axis_min, axis_value)
            span_gain = new_span - current_span
            score += self.QUICK_AXIS_PROGRESS_WEIGHT * span_gain

        # D) Contact with goal borders
        if self.player_id == 1 and (c == 0 or c == n - 1):
            score += self.QUICK_GOAL_BORDER_BONUS

        if self.player_id == 2 and (r == 0 or r == n - 1):
            score += self.QUICK_GOAL_BORDER_BONUS

        return score

    def _best_move_by_dijkstra(self, board: HexBoard, moves: list[Move]) -> Move:
        """
        Among the given candidate moves, choose the best one using the
        full Dijkstra-based position evaluation.

        For each move:
        - simulate the move,
        - evaluate resulting position,
        - keep the highest-scoring move.

        Tie-break:
        - if scores tie, prefer the more central move,
        - then lexicographically smaller coordinate for determinism.

        Args:
            board: Current board.
            moves: Candidate moves to compare.

        Returns:
            Best move among the candidates.
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
                if center_penalty < best_center_penalty:
                    best_move = move
                    best_center_penalty = center_penalty
                elif center_penalty == best_center_penalty and move < best_move:
                    best_move = move

        return best_move

    def _evaluate_position(self, mat: list[list[int]]) -> int:
        """
        Evaluate a board position from our perspective.

        Evaluation:
            score = opp_dist - my_dist

        where:
            my_dist  = shortest connection distance for us
            opp_dist = shortest connection distance for the opponent

        Larger score is better for us.

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
        using multi-source Dijkstra on the hex grid.

        Cell costs from `pid`'s perspective:
            own stone   -> 0
            empty       -> 1
            opponent    -> INF

        So this distance estimates how many empty cells are still needed,
        approximately, to connect the two target borders.

        For player 1:
            start border = left
            goal border  = right

        For player 2:
            start border = top
            goal border  = bottom

        Args:
            mat: Board matrix.
            pid: Player whose distance is being measured.

        Returns:
            Minimum connection cost, or INF if effectively blocked.
        """
        n = len(mat)
        dist = [[self.INF] * n for _ in range(n)]
        pq: list[tuple[int, int, int]] = []

        # Initialize all start-border cells as sources.
        for r, c in self._start_border_cells(n, pid):
            start_cost = self._cell_cost(mat[r][c], pid)

            if start_cost >= self.INF:
                continue

            if start_cost < dist[r][c]:
                dist[r][c] = start_cost
                heapq.heappush(pq, (start_cost, r, c))

        if not pq:
            return self.INF

        # Standard Dijkstra over the hex-cell graph.
        while pq:
            cur_dist, r, c = heapq.heappop(pq)

            if cur_dist != dist[r][c]:
                continue

            # First popped goal-border cell gives the minimum path cost.
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
        Return the traversal cost of a board cell from player `pid`'s view.

        Rules:
            own stone   -> 0
            empty cell  -> 1
            opponent    -> INF

        Args:
            cell_value: Value stored in the board.
            pid: Player perspective.

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
        Check whether (r, c) belongs to the goal border for player `pid`.

        Player 1: right border
        Player 2: bottom border

        Args:
            r: Row index.
            c: Column index.
            n: Board size.
            pid: Player ID.

        Returns:
            True if the cell is on the goal border, otherwise False.
        """
        if pid == 1:
            return c == n - 1
        return r == n - 1

    def _current_goal_axis_bounds(self, mat: list[list[int]]) -> tuple[int | None, int | None]:
        """
        Compute the current min/max occupied coordinate for our stones
        along the axis we want to connect.

        For player 1:
            goal axis = columns

        For player 2:
            goal axis = rows

        Args:
            mat: Board matrix.

        Returns:
            (min_value, max_value), or (None, None) if we have no stones yet.
        """
        n = len(mat)
        values: list[int] = []

        for r in range(n):
            for c in range(n):
                if mat[r][c] == self.player_id:
                    values.append(c if self.player_id == 1 else r)

        if not values:
            return None, None

        return min(values), max(values)

    def _neighbors_evenr(self, n: int, r: int, c: int) -> list[Move]:
        """
        Return the valid hex neighbors of (r, c) using even-r layout.

        Even rows and odd rows use different offsets in this coordinate system.

        Args:
            n: Board size.
            r: Row index.
            c: Column index.

        Returns:
            List of valid neighboring coordinates.
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
        Tie-break helper: prefer more central moves when evaluation ties.

        Lower penalty is better.

        Args:
            mat: Board matrix.
            move: Candidate move.

        Returns:
            Squared distance to the board center.
        """
        n = len(mat)
        r, c = move
        center = (n - 1) / 2.0
        return (r - center) ** 2 + (c - center) ** 2