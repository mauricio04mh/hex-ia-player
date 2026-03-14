from __future__ import annotations

"""
========================================================
GENERAL IDEA
========================================================
This player follows the first solid improvement step for a HEX agent:

1) Win-now:
   If there is any move that wins immediately, play it.

2) Block-now:
   If the opponent has any move that wins immediately on their next turn,
   block that move.

3) Fallback heuristic:
   If there is no forced tactical move, evaluate legal moves with a simple
   heuristic based on:
   - centrality,
   - local connectivity to our stones,
   - local contact with opponent stones,
   - progress along our goal axis,
   - touching our target borders.

4) Safety filter:
   Among the top scored moves, prefer one that does NOT allow the opponent
   to win immediately on the next move.

========================================================
WHY THIS IS A GOOD V1
========================================================
- It is much stronger than random play.
- It avoids the most obvious tactical blunders.

========================================================
IMPORTANT NOTES
========================================================
- This is still a shallow player: it only detects immediate wins/losses.
- It does not yet have deep strategy or search.
"""

from board import HexBoard
from player import Player


Move = tuple[int, int]


class SmartPlayer(Player):
    """
    Rule-based baseline player (Iteration 1).

    Priority order:
    1) Win immediately if possible.
    2) Block opponent's immediate win if necessary.
    3) Otherwise choose a move using a simple static heuristic.
    4) Prefer safe moves that do not allow an opponent win in one.
    """

    # Number of top heuristic moves that will be checked by the safety filter.
    SAFE_TOP_K = 8

    # Heuristic weights for the fallback evaluation.
    CENTER_WEIGHT = -1.0
    OWN_NEIGHBOR_WEIGHT = 2.5
    OPP_NEIGHBOR_WEIGHT = 0.8
    AXIS_PROGRESS_WEIGHT = 3.0
    GOAL_BORDER_BONUS = 2.0

    def __init__(self, player_id: int):
        """
        Initialize the player.

        Args:
            player_id: 1 if this player connects left-right,
                       2 if this player connects top-bottom.
        """
        super().__init__(player_id)
        self.opponent_id = 2 if player_id == 1 else 1

    def play(self, board: HexBoard) -> Move:
        """
        Select the move to play on the current board.

        Decision order:
        1) If we can win immediately, do it.
        2) If the opponent can win immediately, block them.
        3) Otherwise rank all legal moves by heuristic score.
        4) Among the top moves, prefer one that does not allow an
           immediate opponent win on the next turn.

        Args:
            board: Current HEX board (copy provided by the game engine).

        Returns:
            A tuple (row, col) representing the chosen move.
        """
        legal_moves = self._legal_moves(board.board)
        if not legal_moves:
            raise ValueError("No legal moves available.")

        # ------------------------------------------------------------
        # 1) Tactical priority: if we can win in one move, do it now.
        # ------------------------------------------------------------
        for move in legal_moves:
            if self._wins_after_move(board, move, self.player_id):
                return move

        # ------------------------------------------------------------
        # 2) Tactical defense: if the opponent can win in one move,
        #    block that winning cell.
        #
        # If there are multiple forced blocks, we rank them and prefer
        # one that does not still leave an immediate winning reply.
        # ------------------------------------------------------------
        forced_blocks = [
            move
            for move in legal_moves
            if self._wins_after_move(board, move, self.opponent_id)
        ]

        if forced_blocks:
            ranked_blocks = self._rank_moves_by_score(board.board, forced_blocks)

            # Prefer a forced block that does not still allow an
            # immediate opponent win afterward.
            for move in ranked_blocks:
                if not self._allows_opponent_win_in_one(board, move):
                    return move

            # If all blocks still lose tactically, choose the best-scored one.
            return ranked_blocks[0]

        # ------------------------------------------------------------
        # 3) No immediate tactics: rank all moves using the fallback
        #    heuristic.
        # ------------------------------------------------------------
        ranked_moves = self._rank_moves_by_score(board.board, legal_moves)

        # ------------------------------------------------------------
        # 4) Safety filter:
        #    Among the top heuristic moves, prefer one that does not
        #    allow the opponent to win immediately next turn.
        #
        # This is not "deep search", but it makes the baseline more
        # robust in practice.
        # ------------------------------------------------------------
        top_k = min(self.SAFE_TOP_K, len(ranked_moves))
        for move in ranked_moves[:top_k]:
            if not self._allows_opponent_win_in_one(board, move):
                return move

        # If all top moves are tactically bad, return the best heuristic move.
        return ranked_moves[0]

    def _legal_moves(self, mat: list[list[int]]) -> list[Move]:
        """
        Return all empty cells on the board.

        Args:
            mat: NxN matrix of the board.

        Returns:
            List of moves (row, col) where the cell is empty.
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
        Check whether a given player would win immediately by playing `move`.

        This function clones the board, applies the move, and then asks
        the board whether that player has connected their two goal sides.

        Args:
            board: Current game board.
            move: Candidate move (row, col).
            pid: Player ID to test (1 or 2).

        Returns:
            True if placing the piece leads to an immediate win, else False.
        """
        sim = board.clone()

        if not sim.place_piece(move[0], move[1], pid):
            return False

        return sim.check_connection(pid)

    def _allows_opponent_win_in_one(self, board: HexBoard, my_move: Move) -> bool:
        """
        Check whether our move gives the opponent an immediate winning reply.

        Procedure:
        1) Simulate our move.
        2) Enumerate opponent legal replies.
        3) If any opponent reply wins immediately, return True.

        Args:
            board: Current game board.
            my_move: Move we are considering for ourselves.

        Returns:
            True if the opponent can win in one move after `my_move`,
            otherwise False.
        """
        sim = board.clone()

        # If for any reason our own move cannot be played, treat it as bad.
        if not sim.place_piece(my_move[0], my_move[1], self.player_id):
            return True

        for opp_move in self._legal_moves(sim.board):
            if self._wins_after_move(sim, opp_move, self.opponent_id):
                return True

        return False

    def _rank_moves_by_score(self, mat: list[list[int]], moves: list[Move]) -> list[Move]:
        """
        Sort moves from best to worst according to the static heuristic.

        To avoid recomputing the player's current axis span for every move,
        we compute the current axis bounds once here and pass them to the
        scoring function.

        Args:
            mat: Current board matrix.
            moves: Candidate moves to rank.

        Returns:
            A new list of moves sorted from best to worst.
        """
        axis_min, axis_max = self._current_goal_axis_bounds(mat)

        return sorted(
            moves,
            key=lambda m: (
                -self._score_move(mat, m, axis_min, axis_max),
                m[0],
                m[1],
            ),
        )

    def _score_move(
        self,
        mat: list[list[int]],
        move: Move,
        axis_min: int | None,
        axis_max: int | None,
    ) -> float:
        """
        Compute the fallback heuristic score for a move.

        Features used:
        A) Centrality:
           Prefer moves closer to the center of the board.

        B) Local connectivity:
           Reward moves adjacent to our own stones.
           Give a smaller bonus for adjacency to opponent stones because
           contested areas are often relevant.

        C) Goal-axis progress:
           Reward moves that expand our current span in the direction we
           need to connect:
           - Player 1 wants to expand across columns (left-right).
           - Player 2 wants to expand across rows (top-bottom).

        D) Goal-border contact:
           Give a small bonus if the move touches one of the player's
           target borders.

        Args:
            mat: Current board matrix.
            move: Candidate move (row, col).
            axis_min: Current minimum occupied coordinate along the player's
                      goal axis, or None if the player has no stones yet.
            axis_max: Current maximum occupied coordinate along the player's
                      goal axis, or None if the player has no stones yet.

        Returns:
            Numeric score: higher is better.
        """
        n = len(mat)
        r, c = move
        score = 0.0

        # ------------------------------------------------------------
        # A) Centrality:
        #    Moves closer to the center tend to be more flexible in HEX.
        # ------------------------------------------------------------
        center = (n - 1) / 2.0
        dist2 = (r - center) ** 2 + (c - center) ** 2
        score += self.CENTER_WEIGHT * dist2

        # ------------------------------------------------------------
        # B) Local connectivity:
        #    Reward touching own stones, and mildly reward playing in
        #    contested spaces near the opponent.
        # ------------------------------------------------------------
        own_neighbors = 0
        opp_neighbors = 0

        for nr, nc in self._neighbors_evenr(n, r, c):
            cell = mat[nr][nc]
            if cell == self.player_id:
                own_neighbors += 1
            elif cell == self.opponent_id:
                opp_neighbors += 1

        score += self.OWN_NEIGHBOR_WEIGHT * own_neighbors
        score += self.OPP_NEIGHBOR_WEIGHT * opp_neighbors

        # ------------------------------------------------------------
        # C) Goal-axis progress:
        #    Player 1 connects left-right, so columns matter.
        #    Player 2 connects top-bottom, so rows matter.
        #
        #    We reward moves that increase our current covered span on
        #    that target axis.
        # ------------------------------------------------------------
        axis_value = c if self.player_id == 1 else r

        if axis_min is not None and axis_max is not None:
            current_span = axis_max - axis_min
            new_span = max(axis_max, axis_value) - min(axis_min, axis_value)
            span_gain = new_span - current_span
            score += self.AXIS_PROGRESS_WEIGHT * span_gain

        # ------------------------------------------------------------
        # D) Goal-border contact:
        #    Small bonus for directly touching a target border.
        # ------------------------------------------------------------
        if self.player_id == 1 and (c == 0 or c == n - 1):
            score += self.GOAL_BORDER_BONUS

        if self.player_id == 2 and (r == 0 or r == n - 1):
            score += self.GOAL_BORDER_BONUS

        return score

    def _current_goal_axis_bounds(self, mat: list[list[int]]) -> tuple[int | None, int | None]:
        """
        Compute the current min/max occupied coordinate for this player on
        their goal axis.

        For player 1:
            goal axis = columns (left-right connection)

        For player 2:
            goal axis = rows (top-bottom connection)

        Args:
            mat: Current board matrix.

        Returns:
            (min_value, max_value) on the goal axis, or (None, None) if the
            player has no stones on the board yet.
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
        Return the valid HEX neighbors of cell (r, c) using the even-r layout.

        In even-r offset coordinates:
        - even rows and odd rows use different neighbor deltas.

        Args:
            n: Board size.
            r: Row index.
            c: Column index.

        Returns:
            List of valid neighbor coordinates.
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

        neighbors: list[Move] = []
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                neighbors.append((nr, nc))

        return neighbors