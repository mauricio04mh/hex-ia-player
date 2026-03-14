from __future__ import annotations

"""
solution.py

SmartPlayer V4.1 for HEX
========================

This version improves Player 4 by focusing on a smarter candidate selector.

Main upgrades over V4:
1) Better Top-K selector
   - Candidate generation is stronger and safer.

2) Two-stage ranking
   - Stage 1: cheap heuristic ranking over all legal moves.
   - Stage 2: stronger reranking on a medium shortlist using:
       * Dijkstra-based positional score
       * tactical danger penalty
       * local structure bonuses

3) Dynamic K
   - The final number of kept candidates adapts to:
       * board size
       * game phase (how full the board is)

4) Group-merge bonus
   - Moves that connect multiple friendly components are rewarded.

5) Bridge-like bonus
   - Moves that create or strengthen local bridge-style virtual
     connections are rewarded.

Search core:
- immediate win-now / block-now rules
- alpha-beta pruning
- iterative deepening
- shortest-path evaluation at leaves

Assumed board API:
- board.board -> matrix of ints
- board.clone()
- board.place_piece(r, c, pid)
- board.check_connection(pid)

Tunable parameters:
- base_top_k: base candidate budget
- max_depth: iterative deepening max depth
- time_limit_s: soft per-move time cutoff
- prefilter_multiplier: size of the stage-1 shortlist relative to K
"""

import heapq
import time
from collections import deque

from board import HexBoard
from player import Player


Move = tuple[int, int]


class TimeUp(Exception):
    """Internal exception used to stop search when time is up."""
    pass


class SmartPlayer(Player):
    """
    Player 4.1:
    stronger candidate generation + alpha-beta + iterative deepening.
    """

    INF = 10**9
    WIN_SCORE = 10**7

    # -----------------------------------------------------------------
    # Stage-1 quick heuristic weights
    # -----------------------------------------------------------------
    QUICK_CENTER_WEIGHT = -1.0
    QUICK_OWN_NEIGHBOR_WEIGHT = 2.5
    QUICK_OPP_NEIGHBOR_WEIGHT = 1.0
    QUICK_AXIS_PROGRESS_WEIGHT = 3.0
    QUICK_GOAL_BORDER_BONUS = 2.0
    QUICK_GROUP_MERGE_WEIGHT = 4.0
    QUICK_BRIDGE_WEIGHT = 2.5

    # -----------------------------------------------------------------
    # Stage-2 reranking weights
    # -----------------------------------------------------------------
    RERANK_GROUP_MERGE_WEIGHT = 2.0
    RERANK_BRIDGE_WEIGHT = 1.5
    RERANK_DANGER_PENALTY = 5000

    def __init__(
        self,
        player_id: int,
        base_top_k: int = 7,
        max_depth: int = 3,
        time_limit_s: float = 4.0,
        prefilter_multiplier: int = 3,
    ):
        """
        Initialize the player.

        Args:
            player_id:
                1 -> connects left to right
                2 -> connects top to bottom

            base_top_k:
                Base candidate budget. The actual K used per node is dynamic.

            max_depth:
                Maximum iterative deepening depth.

            time_limit_s:
                Soft internal time limit.

            prefilter_multiplier:
                Stage-1 shortlist size is roughly dynamic_K * this multiplier.
        """
        super().__init__(player_id)
        self.opponent_id = 2 if player_id == 1 else 1

        self.base_top_k = base_top_k
        self.max_depth = max_depth
        self.time_limit_s = time_limit_s
        self.prefilter_multiplier = max(2, prefilter_multiplier)

        self._search_start_time = 0.0

    # ============================================================
    # Public entry point
    # ============================================================

    def play(self, board: HexBoard) -> Move:
        """
        Choose the move for the current position.

        Order:
        1) Immediate tactical win.
        2) Immediate tactical block.
        3) Generate improved root candidates.
        4) Run iterative deepening alpha-beta on those candidates.
        5) Return best move from the deepest completed depth.
        """
        self._search_start_time = time.perf_counter()

        legal_moves = self._legal_moves(board.board)
        if not legal_moves:
            raise ValueError("No legal moves available.")

        # 1) Win now if possible.
        for move in legal_moves:
            if self._wins_after_move(board, move, self.player_id):
                return move

        # 2) Block opponent immediate win if necessary.
        forced_blocks = [
            move
            for move in legal_moves
            if self._wins_after_move(board, move, self.opponent_id)
        ]
        if forced_blocks:
            return self._best_move_by_dijkstra(board, forced_blocks)

        # 3) Improved candidate generation for the root.
        root_moves = self._candidate_moves(board, self.player_id)
        if not root_moves:
            return legal_moves[0]

        best_move = root_moves[0]

        # 4) Iterative deepening.
        for depth in range(1, self.max_depth + 1):
            try:
                score, move = self._alpha_beta_root(board, depth, root_moves)
                best_move = move

                # PV-style ordering: best move first next iteration.
                root_moves = [move] + [m for m in root_moves if m != move]

            except TimeUp:
                break

        return best_move

    # ============================================================
    # Alpha-beta search
    # ============================================================

    def _alpha_beta_root(self, board: HexBoard, depth: int, root_moves: list[Move]) -> tuple[int, Move]:
        """
        Root alpha-beta search for a fixed depth.
        """
        self._check_time()

        alpha = -self.INF
        beta = self.INF

        best_score = -self.INF
        best_move = root_moves[0]

        for move in root_moves:
            self._check_time()

            sim = board.clone()
            if not sim.place_piece(move[0], move[1], self.player_id):
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
        Returns evaluation from OUR perspective.
        """
        self._check_time()

        # Terminal positions.
        if board.check_connection(self.player_id):
            return self.WIN_SCORE + depth

        if board.check_connection(self.opponent_id):
            return -self.WIN_SCORE - depth

        if depth == 0:
            return self._evaluate_position(board.board)

        legal_moves = self._legal_moves(board.board)
        if not legal_moves:
            return self._evaluate_position(board.board)

        candidate_moves = self._candidate_moves(board, current_pid)
        if not candidate_moves:
            return self._evaluate_position(board.board)

        if current_pid == self.player_id:
            value = -self.INF

            for move in candidate_moves:
                self._check_time()

                sim = board.clone()
                if not sim.place_piece(move[0], move[1], current_pid):
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
                if not sim.place_piece(move[0], move[1], current_pid):
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
        Raise TimeUp if the soft time limit has been reached.
        """
        if time.perf_counter() - self._search_start_time >= self.time_limit_s:
            raise TimeUp()

    # ============================================================
    # Candidate generation (the main 4.1 improvement)
    # ============================================================

    def _candidate_moves(self, board: HexBoard, pid: int) -> list[Move]:
        """
        Generate strong candidate moves for player `pid`.

        Strategy:
        1) Tactical win-now if present.
        2) Tactical block-now if needed.
        3) Dynamic K based on board size and phase.
        4) Stage-1 quick ranking over all legal moves.
        5) Keep a medium shortlist (prefilter).
        6) Stage-2 rerank that shortlist using:
           - simulated Dijkstra score
           - tactical danger penalty
           - local structure bonuses
        7) Return final Top-K.
        """
        legal_moves = self._legal_moves(board.board)
        if not legal_moves:
            return []

        opponent = 2 if pid == 1 else 1

        # Tactical safety: never prune winning moves.
        winning_moves = [
            move
            for move in legal_moves
            if self._wins_after_move(board, move, pid)
        ]
        if winning_moves:
            return self._sort_moves_by_quick_score(board.board, winning_moves, pid)

        # Tactical safety: never prune forced blocks.
        forced_blocks = [
            move
            for move in legal_moves
            if self._wins_after_move(board, move, opponent)
        ]
        if forced_blocks:
            return self._sort_moves_by_quick_score(board.board, forced_blocks, pid)

        dynamic_k = self._dynamic_top_k(board.board, len(legal_moves))
        prefilter_size = min(
            len(legal_moves),
            max(dynamic_k * self.prefilter_multiplier, dynamic_k + 6),
        )

        # Stage 1: cheap ranking over all legal moves.
        quick_ranked = self._sort_moves_by_quick_score(board.board, legal_moves, pid)
        shortlist = quick_ranked[:prefilter_size]

        # Stage 2: stronger reranking over the shortlist.
        strong_ranked = self._rerank_shortlist(board, shortlist, pid)

        return strong_ranked[:dynamic_k]

    def _dynamic_top_k(self, mat: list[list[int]], legal_count: int) -> int:
        """
        Compute a dynamic K based on board size and game phase.

        Intuition:
        - larger boards need wider candidate coverage
        - early game needs wider coverage than late game
        """
        n = len(mat)
        total = n * n
        emptiness = legal_count / total

        k = self.base_top_k

        # Bigger boards usually need a slightly larger candidate budget.
        if n >= 11:
            k += 2
        if n >= 13:
            k += 2

        # Opening / early game: many plausible strategic moves.
        if emptiness > 0.75:
            k += 4
        elif emptiness > 0.55:
            k += 2
        elif emptiness < 0.25:
            k -= 2

        k = max(4, min(k, legal_count))
        return k

    def _sort_moves_by_quick_score(self, mat: list[list[int]], moves: list[Move], pid: int) -> list[Move]:
        """
        Stage-1 quick ranking.

        Uses:
        - centrality
        - local contact
        - axis progress
        - goal-border bonus
        - group-merge bonus
        - bridge-like bonus
        """
        axis_min, axis_max = self._current_goal_axis_bounds(mat, pid)
        component_ids = self._component_id_map(mat, pid)

        return sorted(
            moves,
            key=lambda m: (
                -self._quick_score_move(mat, m, pid, axis_min, axis_max, component_ids),
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
        component_ids: list[list[int]],
    ) -> float:
        """
        Cheap ranking heuristic used in stage 1.

        This is intentionally lightweight but stronger than the old V4
        quick scorer because it also values:
        - joining multiple groups
        - creating bridge-like structure
        """
        n = len(mat)
        r, c = move
        opponent = 2 if pid == 1 else 1

        score = 0.0

        # A) Centrality
        center = (n - 1) / 2.0
        dist2 = (r - center) ** 2 + (c - center) ** 2
        score += self.QUICK_CENTER_WEIGHT * dist2

        # B) Local adjacency / contest
        own_neighbors = 0
        opp_neighbors = 0
        for nr, nc in self._neighbors_evenr(n, r, c):
            if mat[nr][nc] == pid:
                own_neighbors += 1
            elif mat[nr][nc] == opponent:
                opp_neighbors += 1

        score += self.QUICK_OWN_NEIGHBOR_WEIGHT * own_neighbors
        score += self.QUICK_OPP_NEIGHBOR_WEIGHT * opp_neighbors

        # C) Goal-axis progress
        axis_value = c if pid == 1 else r
        if axis_min is not None and axis_max is not None:
            current_span = axis_max - axis_min
            new_span = max(axis_max, axis_value) - min(axis_min, axis_value)
            span_gain = new_span - current_span
            score += self.QUICK_AXIS_PROGRESS_WEIGHT * span_gain

        # D) Goal-border bonus
        if pid == 1 and (c == 0 or c == n - 1):
            score += self.QUICK_GOAL_BORDER_BONUS
        if pid == 2 and (r == 0 or r == n - 1):
            score += self.QUICK_GOAL_BORDER_BONUS

        # E) Group-merge bonus
        merged_components = self._adjacent_friendly_component_count(
            mat, move, pid, component_ids
        )
        if merged_components >= 2:
            score += self.QUICK_GROUP_MERGE_WEIGHT * (merged_components - 1)

        # F) Bridge-like bonus
        score += self.QUICK_BRIDGE_WEIGHT * self._bridge_like_count(mat, move, pid)

        return score

    def _rerank_shortlist(self, board: HexBoard, shortlist: list[Move], pid: int) -> list[Move]:
        """
        Stage-2 strong reranking on a medium shortlist.

        For each move:
        - simulate it
        - evaluate resulting position from that player's perspective
        - penalize moves that allow an immediate opponent win
        - add local group-merge and bridge-like bonuses

        This makes a small K much safer than the old V4.
        """
        opponent = 2 if pid == 1 else 1

        scored: list[tuple[float, Move]] = []
        for move in shortlist:
            self._check_time()

            sim = board.clone()
            if not sim.place_piece(move[0], move[1], pid):
                continue

            score = float(self._evaluate_position_for_pid(sim.board, pid))

            # Tactical danger penalty:
            # after this move, can opponent win immediately?
            if self._has_immediate_winning_reply(sim, opponent):
                score -= self.RERANK_DANGER_PENALTY

            # Local structure bonuses on the simulated position.
            comp_ids_after = self._component_id_map(sim.board, pid)
            merged_components = self._adjacent_friendly_component_count(
                sim.board, move, pid, comp_ids_after
            )
            if merged_components >= 2:
                score += self.RERANK_GROUP_MERGE_WEIGHT * (merged_components - 1)

            score += self.RERANK_BRIDGE_WEIGHT * self._bridge_like_count(sim.board, move, pid)

            # Small tie-break preference toward center.
            score -= 0.001 * self._center_penalty(sim.board, move)

            scored.append((score, move))

        scored.sort(key=lambda x: (-x[0], x[1][0], x[1][1]))
        return [move for _, move in scored]

    # ============================================================
    # Tactical helpers
    # ============================================================

    def _wins_after_move(self, board: HexBoard, move: Move, pid: int) -> bool:
        """
        Check whether player `pid` wins immediately by playing `move`.
        """
        sim = board.clone()
        if not sim.place_piece(move[0], move[1], pid):
            return False
        return sim.check_connection(pid)

    def _has_immediate_winning_reply(self, board: HexBoard, pid: int) -> bool:
        """
        Check whether player `pid` has any immediate winning move.
        """
        for move in self._legal_moves(board.board):
            if self._wins_after_move(board, move, pid):
                return True
        return False

    def _best_move_by_dijkstra(self, board: HexBoard, moves: list[Move]) -> Move:
        """
        Choose the best move among a small tactical set using the
        Dijkstra-based static evaluation.
        """
        best_move = moves[0]
        best_score = -self.INF
        best_center_penalty = self._center_penalty(board.board, best_move)

        for move in moves:
            sim = board.clone()
            if not sim.place_piece(move[0], move[1], self.player_id):
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
        Larger is better for us.
        """
        return self._evaluate_position_for_pid(mat, self.player_id)

    def _evaluate_position_for_pid(self, mat: list[list[int]], pid: int) -> int:
        """
        Evaluate the board from player `pid`'s perspective.

        Score:
            opponent_distance - my_distance
        """
        opponent = 2 if pid == 1 else 1
        my_dist = self._shortest_connection_distance(mat, pid)
        opp_dist = self._shortest_connection_distance(mat, opponent)
        return opp_dist - my_dist

    def _shortest_connection_distance(self, mat: list[list[int]], pid: int) -> int:
        """
        Multi-source Dijkstra shortest-path connectivity estimate.

        Costs from player `pid`'s perspective:
        - own stone -> 0
        - empty     -> 1
        - opponent  -> INF
        """
        n = len(mat)
        dist = [[self.INF] * n for _ in range(n)]
        pq: list[tuple[int, int, int]] = []

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
        """
        if cell_value == pid:
            return 0
        if cell_value == 0:
            return 1
        return self.INF

    # ============================================================
    # Structural helpers: components and bridge-like patterns
    # ============================================================

    def _component_id_map(self, mat: list[list[int]], pid: int) -> list[list[int]]:
        """
        Compute connected-component IDs for all stones belonging to `pid`.

        Empty or opponent cells get -1.
        """
        n = len(mat)
        comp = [[-1] * n for _ in range(n)]
        comp_id = 0

        for r in range(n):
            for c in range(n):
                if mat[r][c] != pid or comp[r][c] != -1:
                    continue

                q = deque([(r, c)])
                comp[r][c] = comp_id

                while q:
                    cr, cc = q.popleft()
                    for nr, nc in self._neighbors_evenr(n, cr, cc):
                        if mat[nr][nc] == pid and comp[nr][nc] == -1:
                            comp[nr][nc] = comp_id
                            q.append((nr, nc))

                comp_id += 1

        return comp

    def _adjacent_friendly_component_count(
        self,
        mat: list[list[int]],
        move: Move,
        pid: int,
        component_ids: list[list[int]],
    ) -> int:
        """
        Count how many DISTINCT friendly components are adjacent to `move`.
        """
        n = len(mat)
        r, c = move
        seen: set[int] = set()

        for nr, nc in self._neighbors_evenr(n, r, c):
            if mat[nr][nc] == pid:
                cid = component_ids[nr][nc]
                if cid != -1:
                    seen.add(cid)

        return len(seen)

    def _bridge_like_count(self, mat: list[list[int]], move: Move, pid: int) -> int:
        """
        Approximate local bridge / virtual-connection patterns.

        Idea:
        - Consider the move as one endpoint.
        - Look for friendly stones at distance 2 (not adjacent).
        - If the move and that stone have at least two common empty neighbors,
          that resembles a bridge-like virtual connection.

        This is an approximation, but it is useful for candidate ranking.
        """
        n = len(mat)
        r, c = move

        direct_neighbors = set(self._neighbors_evenr(n, r, c))
        two_hop: set[Move] = set()

        for nr, nc in direct_neighbors:
            for tr, tc in self._neighbors_evenr(n, nr, nc):
                if (tr, tc) != (r, c):
                    two_hop.add((tr, tc))

        count = 0
        for tr, tc in two_hop:
            if (tr, tc) in direct_neighbors:
                continue
            if mat[tr][tc] != pid:
                continue

            target_neighbors = set(self._neighbors_evenr(n, tr, tc))
            common = direct_neighbors & target_neighbors

            empty_common = 0
            for cr, cc in common:
                if mat[cr][cc] == 0:
                    empty_common += 1

            if empty_common >= 2:
                count += 1

        return count

    # ============================================================
    # Board utilities
    # ============================================================

    def _legal_moves(self, mat: list[list[int]]) -> list[Move]:
        """
        Return all legal moves (empty cells).
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
        Return current min/max occupied coordinate along the player's goal axis.

        Player 1 -> columns matter
        Player 2 -> rows matter
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
        Return all source-border cells for player `pid`.
        """
        if pid == 1:
            return [(r, 0) for r in range(n)]
        return [(0, c) for c in range(n)]

    def _is_goal_border_cell(self, r: int, c: int, n: int, pid: int) -> bool:
        """
        Check whether a cell belongs to the goal border for player `pid`.
        """
        if pid == 1:
            return c == n - 1
        return r == n - 1

    def _neighbors_evenr(self, n: int, r: int, c: int) -> list[Move]:
        """
        Return valid HEX neighbors of (r, c) using even-r layout.
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
        Squared distance to board center, used for tie-breaking.
        """
        n = len(mat)
        r, c = move
        center = (n - 1) / 2.0
        return (r - center) ** 2 + (c - center) ** 2