from __future__ import annotations

"""
solution.py

SmartPlayer V4 - Iteration 4 for HEX
====================================

This player implements:

1) Tactical rules:
   - If we can win immediately, do it.
   - If the opponent can win immediately on their next move, block it.

2) Candidate generation with Top-K pruning:
   - Rank legal moves quickly with a cheap heuristic.
   - Keep only the best K candidate moves.

3) Alpha-beta search:
   - Search deeper than 1-ply.
   - Assume the opponent also plays well.

4) Iterative deepening:
   - Search depth 1, then 2, then 3, ... up to max_depth.
   - Stop early if time is almost exhausted.
   - Return the best move from the last fully completed depth.

5) Leaf evaluation:
   - Use shortest-path / Dijkstra evaluation:
       score = opp_dist - my_dist
   - From a player's perspective:
       own stone -> 0
       empty     -> 1
       opponent  -> INF

Why K and D are parameters:
- top_k controls how many candidate moves survive pruning.
- max_depth controls how deeply alpha-beta searches.

Typical tuning ideas:
- Smaller K -> faster, but riskier pruning
- Larger K  -> slower, but safer
- Smaller D -> faster, shallower search
- Larger D  -> stronger if time allows, but much slower

This version is written for clarity and experimentation.
It is a strong class-project baseline, though not the most optimized possible
version (for example, it uses board cloning instead of apply/undo).
"""

import heapq
import time

from board import HexBoard
from player import Player


Move = tuple[int, int]


class TimeUp(Exception):
    """
    Internal exception used to stop search cleanly when the time budget is reached.
    """
    pass


class SmartPlayer(Player):
    """
    HEX player using:
    - tactical rules
    - Top-K candidate pruning
    - alpha-beta
    - iterative deepening
    - Dijkstra evaluation
    """

    INF = 10**9
    WIN_SCORE = 10**7

    # Quick-ranking heuristic weights.
    QUICK_CENTER_WEIGHT = -1.0
    QUICK_OWN_NEIGHBOR_WEIGHT = 2.5
    QUICK_OPP_NEIGHBOR_WEIGHT = 0.8
    QUICK_AXIS_PROGRESS_WEIGHT = 3.0
    QUICK_GOAL_BORDER_BONUS = 2.0

    def __init__(
        self,
        player_id: int,
        top_k: int = 15,
        max_depth: int = 2,
        time_limit_s: float = 4.5,
    ):
        """
        Initialize the player.

        Args:
            player_id:
                1 if this player connects left-right,
                2 if this player connects top-bottom.

            top_k:
                Number of candidate moves kept after quick ranking.

            max_depth:
                Maximum iterative-deepening depth to try.
                Search will run depths 1, 2, ..., max_depth.

            time_limit_s:
                Soft time limit used internally to stop before the hard match limit.
        """
        super().__init__(player_id)
        self.opponent_id = 2 if player_id == 1 else 1

        self.top_k = top_k
        self.max_depth = max_depth
        self.time_limit_s = time_limit_s

        # Search-time state, initialized inside play().
        self._search_start_time = 0.0

    def play(self, board: HexBoard) -> Move:
        """
        Choose a move for the current board.

        Decision flow:
        1) Check immediate tactical win.
        2) Check immediate tactical block.
        3) Generate Top-K root candidates.
        4) Run iterative deepening alpha-beta up to max_depth or until time runs out.
        5) Return the best move from the deepest completed search.

        Args:
            board: Current HEX board.

        Returns:
            A legal move (row, col).
        """
        self._search_start_time = time.perf_counter()

        legal_moves = self._legal_moves(board.board)
        if not legal_moves:
            raise ValueError("No legal moves available.")

        # ------------------------------------------------------------
        # 1) Tactical win-now at root.
        # ------------------------------------------------------------
        for move in legal_moves:
            if self._wins_after_move(board, move, self.player_id):
                return move

        # ------------------------------------------------------------
        # 2) Tactical block-now at root.
        # ------------------------------------------------------------
        forced_blocks = [
            move
            for move in legal_moves
            if self._wins_after_move(board, move, self.opponent_id)
        ]
        if forced_blocks:
            return self._best_move_by_dijkstra(board, forced_blocks)

        # ------------------------------------------------------------
        # 3) Generate ordered root candidates.
        # ------------------------------------------------------------
        root_moves = self._candidate_moves(board, self.player_id, self.top_k)
        if not root_moves:
            return legal_moves[0]

        best_move = root_moves[0]

        # ------------------------------------------------------------
        # 4) Iterative deepening:
        #    Try increasing depths until time runs out or max_depth is reached.
        # ------------------------------------------------------------
        for depth in range(1, self.max_depth + 1):
            try:
                score, move = self._alpha_beta_root(board, depth, root_moves)
                best_move = move

                # Principal variation style ordering:
                # move the best move to the front for the next depth.
                root_moves = [move] + [m for m in root_moves if m != move]

            except TimeUp:
                break

        return best_move

    # ============================================================
    # Core search functions
    # ============================================================

    def _alpha_beta_root(self, board: HexBoard, depth: int, root_moves: list[Move]) -> tuple[int, Move]:
        """
        Search the root position at a given depth using alpha-beta.

        Args:
            board: Current board.
            depth: Search depth to complete.
            root_moves: Ordered candidate moves for the root.

        Returns:
            (best_score, best_move) from the root.
        """
        self._check_time()

        alpha = -self.INF
        beta = self.INF

        best_score = -self.INF
        best_move = root_moves[0]

        for move in root_moves:
            self._check_time()

            sim = board.clone()
            ok = sim.place_piece(move[0], move[1], self.player_id)
            if not ok:
                continue

            score = self._alpha_beta(
                sim,
                depth - 1,
                alpha,
                beta,
                self.opponent_id,
            )

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                break

        return best_score, best_move

    def _alpha_beta(
        self,
        board: HexBoard,
        depth: int,
        alpha: int,
        beta: int,
        current_pid: int,
    ) -> int:
        """
        Recursive alpha-beta search.

        Args:
            board: Current simulated board.
            depth: Remaining plies to search.
            alpha: Alpha bound.
            beta: Beta bound.
            current_pid: Player to move in this node.

        Returns:
            Evaluation score from this node from OUR perspective.
        """
        self._check_time()

        # ------------------------------------------------------------
        # Terminal checks.
        # ------------------------------------------------------------
        if board.check_connection(self.player_id):
            return self.WIN_SCORE + depth

        if board.check_connection(self.opponent_id):
            return -self.WIN_SCORE - depth

        if depth == 0:
            return self._evaluate_position(board.board)

        legal_moves = self._legal_moves(board.board)
        if not legal_moves:
            return self._evaluate_position(board.board)

        candidate_moves = self._candidate_moves(board, current_pid, self.top_k)
        if not candidate_moves:
            return self._evaluate_position(board.board)

        # ------------------------------------------------------------
        # Maximizing node: our turn.
        # Minimizing node: opponent turn.
        # ------------------------------------------------------------
        if current_pid == self.player_id:
            value = -self.INF

            for move in candidate_moves:
                self._check_time()

                sim = board.clone()
                ok = sim.place_piece(move[0], move[1], current_pid)
                if not ok:
                    continue

                child_value = self._alpha_beta(
                    sim,
                    depth - 1,
                    alpha,
                    beta,
                    self.opponent_id,
                )

                if child_value > value:
                    value = child_value

                if value > alpha:
                    alpha = value

                if alpha >= beta:
                    break

            return value

        else:
            value = self.INF

            for move in candidate_moves:
                self._check_time()

                sim = board.clone()
                ok = sim.place_piece(move[0], move[1], current_pid)
                if not ok:
                    continue

                child_value = self._alpha_beta(
                    sim,
                    depth - 1,
                    alpha,
                    beta,
                    self.player_id,
                )

                if child_value < value:
                    value = child_value

                if value < beta:
                    beta = value

                if alpha >= beta:
                    break

            return value

    def _check_time(self) -> None:
        """
        Stop the search if the configured soft time limit has been reached.
        """
        if time.perf_counter() - self._search_start_time >= self.time_limit_s:
            raise TimeUp()

    # ============================================================
    # Candidate generation
    # ============================================================

    def _candidate_moves(self, board: HexBoard, pid: int, top_k: int) -> list[Move]:
        """
        Generate candidate moves for a given player.

        Priority:
        1) If that player can win immediately, return those winning moves.
        2) If the opponent can win immediately next turn, return blocking moves.
        3) Otherwise, rank moves with a cheap heuristic and keep Top-K.

        This is the safer, clearer version for Iteration 4.
        It is not the cheapest possible candidate generator, but it follows
        the roadmap logic closely.

        Args:
            board: Current board.
            pid: Player for whom we are generating candidates.
            top_k: Number of non-forced moves to keep.

        Returns:
            Ordered list of candidate moves.
        """
        legal_moves = self._legal_moves(board.board)
        if not legal_moves:
            return []

        opponent = 2 if pid == 1 else 1

        # 1) Immediate wins for the side to move.
        winning_moves = [
            move
            for move in legal_moves
            if self._wins_after_move(board, move, pid)
        ]
        if winning_moves:
            return self._sort_moves_by_quick_score(board.board, winning_moves, pid)

        # 2) Immediate blocks against opponent wins.
        forced_blocks = [
            move
            for move in legal_moves
            if self._wins_after_move(board, move, opponent)
        ]
        if forced_blocks:
            return self._sort_moves_by_quick_score(board.board, forced_blocks, pid)

        # 3) Otherwise use cheap ranking and Top-K pruning.
        ranked = self._sort_moves_by_quick_score(board.board, legal_moves, pid)
        return ranked[: min(top_k, len(ranked))]

    def _sort_moves_by_quick_score(self, mat: list[list[int]], moves: list[Move], pid: int) -> list[Move]:
        """
        Sort moves using the quick heuristic for the given player.

        Args:
            mat: Current board matrix.
            moves: Moves to sort.
            pid: Perspective player for the quick heuristic.

        Returns:
            Moves ordered from best to worst according to the quick score.
        """
        axis_min, axis_max = self._current_goal_axis_bounds(mat, pid)

        return sorted(
            moves,
            key=lambda m: (
                -self._quick_score_move(mat, m, pid, axis_min, axis_max),
                m[0],
                m[1],
            ),
        )

    def _quick_score_move(
        self,
        mat: list[list[int]],
        move: Move,
        pid: int,
        axis_min: int | None,
        axis_max: int | None,
    ) -> float:
        """
        Cheap move-ranking heuristic.

        Features:
        - centrality
        - adjacency to own stones
        - adjacency to opponent stones
        - expansion along the player's target axis
        - contact with target borders

        Args:
            mat: Board matrix.
            move: Candidate move.
            pid: Perspective player.
            axis_min: Current minimum occupied value on the player's goal axis.
            axis_max: Current maximum occupied value on the player's goal axis.

        Returns:
            Quick heuristic score. Higher is better.
        """
        n = len(mat)
        r, c = move
        opponent = 2 if pid == 1 else 1

        score = 0.0

        # A) Centrality.
        center = (n - 1) / 2.0
        dist2 = (r - center) ** 2 + (c - center) ** 2
        score += self.QUICK_CENTER_WEIGHT * dist2

        # B) Local contact.
        own_neighbors = 0
        opp_neighbors = 0

        for nr, nc in self._neighbors_evenr(n, r, c):
            cell = mat[nr][nc]
            if cell == pid:
                own_neighbors += 1
            elif cell == opponent:
                opp_neighbors += 1

        score += self.QUICK_OWN_NEIGHBOR_WEIGHT * own_neighbors
        score += self.QUICK_OPP_NEIGHBOR_WEIGHT * opp_neighbors

        # C) Goal-axis progress.
        axis_value = c if pid == 1 else r
        if axis_min is not None and axis_max is not None:
            current_span = axis_max - axis_min
            new_span = max(axis_max, axis_value) - min(axis_min, axis_value)
            span_gain = new_span - current_span
            score += self.QUICK_AXIS_PROGRESS_WEIGHT * span_gain

        # D) Goal-border bonus.
        if pid == 1 and (c == 0 or c == n - 1):
            score += self.QUICK_GOAL_BORDER_BONUS

        if pid == 2 and (r == 0 or r == n - 1):
            score += self.QUICK_GOAL_BORDER_BONUS

        return score

    # ============================================================
    # Tactical helpers
    # ============================================================

    def _wins_after_move(self, board: HexBoard, move: Move, pid: int) -> bool:
        """
        Check whether player `pid` wins immediately by playing `move`.

        Args:
            board: Current board.
            move: Candidate move.
            pid: Player to test.

        Returns:
            True if that move produces an immediate win.
        """
        sim = board.clone()
        if not sim.place_piece(move[0], move[1], pid):
            return False
        return sim.check_connection(pid)

    def _best_move_by_dijkstra(self, board: HexBoard, moves: list[Move]) -> Move:
        """
        Choose the best move among a small set of candidates using the
        Dijkstra-based evaluation after simulating our move.

        This is mainly used for root tactical block situations.

        Args:
            board: Current board.
            moves: Candidate moves.

        Returns:
            Best move according to the static evaluation.
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

    # ============================================================
    # Position evaluation
    # ============================================================

    def _evaluate_position(self, mat: list[list[int]]) -> int:
        """
        Evaluate a board position from OUR perspective.

        Evaluation:
            score = opp_dist - my_dist

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
        Compute the shortest connection distance for player `pid` using
        multi-source Dijkstra.

        Costs from `pid`'s perspective:
            own stone -> 0
            empty     -> 1
            opponent  -> INF

        For player 1:
            start border = left
            goal border  = right

        For player 2:
            start border = top
            goal border  = bottom

        Args:
            mat: Board matrix.
            pid: Player being evaluated.

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

        while pq:
            cur_dist, r, c = heapq.heappop(pq)

            if cur_dist != dist[r][c]:
                continue

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
        Return traversal cost of a cell from player `pid`'s perspective.

        Args:
            cell_value: Value stored in a board cell.
            pid: Perspective player.

        Returns:
            0 for own stone, 1 for empty, INF for opponent stone.
        """
        if cell_value == pid:
            return 0
        if cell_value == 0:
            return 1
        return self.INF

    # ============================================================
    # Board utility helpers
    # ============================================================

    def _legal_moves(self, mat: list[list[int]]) -> list[Move]:
        """
        Return all legal moves (empty cells).

        Args:
            mat: Board matrix.

        Returns:
            List of legal moves.
        """
        moves: list[Move] = []
        n = len(mat)

        for r in range(n):
            for c in range(n):
                if mat[r][c] == 0:
                    moves.append((r, c))

        return moves

    def _current_goal_axis_bounds(self, mat: list[list[int]], pid: int) -> tuple[int | None, int | None]:
        """
        Compute the min/max occupied coordinate for `pid` along that player's
        goal axis.

        Player 1:
            goal axis = columns

        Player 2:
            goal axis = rows

        Args:
            mat: Board matrix.
            pid: Player ID.

        Returns:
            (min_value, max_value), or (None, None) if the player has no stones.
        """
        n = len(mat)
        values: list[int] = []

        for r in range(n):
            for c in range(n):
                if mat[r][c] == pid:
                    values.append(c if pid == 1 else r)

        if not values:
            return None, None

        return min(values), max(values)

    def _start_border_cells(self, n: int, pid: int) -> list[Move]:
        """
        Return all cells on the player's starting border.

        Args:
            n: Board size.
            pid: Player ID.

        Returns:
            Border cells for multi-source Dijkstra initialization.
        """
        if pid == 1:
            return [(r, 0) for r in range(n)]
        return [(0, c) for c in range(n)]

    def _is_goal_border_cell(self, r: int, c: int, n: int, pid: int) -> bool:
        """
        Check whether a cell belongs to the player's goal border.

        Args:
            r: Row index.
            c: Column index.
            n: Board size.
            pid: Player ID.

        Returns:
            True if the cell is on the goal border.
        """
        if pid == 1:
            return c == n - 1
        return r == n - 1

    def _neighbors_evenr(self, n: int, r: int, c: int) -> list[Move]:
        """
        Return the valid HEX neighbors of (r, c) using even-r layout.

        Even rows and odd rows use different neighbor offsets.

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
        Tie-break helper: prefer more central moves when scores tie.

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