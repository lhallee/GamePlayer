from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from game_player.games.base import Action, Player

BLACK: Player = 1
WHITE: Player = -1
EMPTY: Player = 0

Board = tuple[Player, ...]


def other_player(player: Player) -> Player:
    assert player in (BLACK, WHITE)
    return WHITE if player == BLACK else BLACK


def point_to_action(row: int, col: int, board_size: int) -> Action:
    assert 0 <= row < board_size
    assert 0 <= col < board_size
    return row * board_size + col


def action_to_point(action: Action, board_size: int) -> tuple[int, int]:
    assert 0 <= action < board_size * board_size
    return divmod(action, board_size)


def pass_action(board_size: int) -> Action:
    return board_size * board_size


def _neighbors(point: int, board_size: int) -> Iterable[int]:
    row, col = divmod(point, board_size)
    if row > 0:
        yield point - board_size
    if row + 1 < board_size:
        yield point + board_size
    if col > 0:
        yield point - 1
    if col + 1 < board_size:
        yield point + 1


def _collect_group(board: Board, start: int, board_size: int) -> tuple[set[int], set[int]]:
    color = board[start]
    assert color != EMPTY

    group = {start}
    liberties: set[int] = set()
    frontier = [start]

    while frontier:
        point = frontier.pop()
        for neighbor in _neighbors(point, board_size):
            neighbor_color = board[neighbor]
            if neighbor_color == EMPTY:
                liberties.add(neighbor)
            elif neighbor_color == color and neighbor not in group:
                group.add(neighbor)
                frontier.append(neighbor)

    return group, liberties


def _collect_empty_region(
    board: Board,
    start: int,
    board_size: int,
) -> tuple[set[int], set[Player]]:
    assert board[start] == EMPTY

    region = {start}
    border_colors: set[Player] = set()
    frontier = [start]

    while frontier:
        point = frontier.pop()
        for neighbor in _neighbors(point, board_size):
            neighbor_color = board[neighbor]
            if neighbor_color == EMPTY and neighbor not in region:
                region.add(neighbor)
                frontier.append(neighbor)
            elif neighbor_color != EMPTY:
                border_colors.add(neighbor_color)

    return region, border_colors


@dataclass(frozen=True)
class GoState:
    board_size: int
    komi: float
    board: Board
    current_player: Player
    consecutive_passes: int
    position_history: tuple[Board, ...]
    position_history_players: tuple[Player, ...] = ()

    @classmethod
    def new(cls, board_size: int = 19, komi: float = 7.5) -> "GoState":
        assert board_size > 1
        board = (EMPTY,) * (board_size * board_size)
        return cls(
            board_size=board_size,
            komi=komi,
            board=board,
            current_player=BLACK,
            consecutive_passes=0,
            position_history=(board,),
            position_history_players=(EMPTY,),
        )

    @property
    def action_size(self) -> int:
        return self.board_size * self.board_size + 1

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return 2, self.board_size, self.board_size

    @property
    def pass_action(self) -> Action:
        return pass_action(self.board_size)

    def is_terminal(self) -> bool:
        return self.consecutive_passes >= 2

    def is_legal_action(self, action: Action) -> bool:
        if self.is_terminal():
            return False
        if action == self.pass_action:
            return True
        if not 0 <= action < self.board_size * self.board_size:
            return False
        if self.board[action] != EMPTY:
            return False
        next_board = self._next_board_for_play(action)
        if self._player_has_left_position(self.current_player, next_board):
            return False
        return True

    def legal_actions(self) -> list[Action]:
        if self.is_terminal():
            return []
        actions = [
            action
            for action in range(self.board_size * self.board_size)
            if self.is_legal_action(action)
        ]
        actions.append(self.pass_action)
        return actions

    def apply_action(self, action: Action) -> "GoState":
        assert self.is_legal_action(action)

        if action == self.pass_action:
            return GoState(
                board_size=self.board_size,
                komi=self.komi,
                board=self.board,
                current_player=other_player(self.current_player),
                consecutive_passes=self.consecutive_passes + 1,
                position_history=self.position_history + (self.board,),
                position_history_players=(
                    self._history_players() + (self.current_player,)
                ),
            )

        next_board = self._next_board_for_play(action)
        return GoState(
            board_size=self.board_size,
            komi=self.komi,
            board=next_board,
            current_player=other_player(self.current_player),
            consecutive_passes=0,
            position_history=self.position_history + (next_board,),
            position_history_players=(
                self._history_players() + (self.current_player,)
            ),
        )

    def _next_board_for_play(self, action: Action) -> Board:
        next_board = list(self.board)
        next_board[action] = self.current_player
        opponent = other_player(self.current_player)

        for neighbor in _neighbors(action, self.board_size):
            if next_board[neighbor] != opponent:
                continue
            group, liberties = _collect_group(tuple(next_board), neighbor, self.board_size)
            if liberties:
                continue
            for point in group:
                next_board[point] = EMPTY

        board_after_captures = tuple(next_board)
        own_group, own_liberties = _collect_group(
            board_after_captures,
            action,
            self.board_size,
        )
        assert own_group
        if not own_liberties:
            for point in own_group:
                next_board[point] = EMPTY
            return tuple(next_board)
        return board_after_captures

    def _history_players(self) -> tuple[Player, ...]:
        if self.position_history_players:
            assert len(self.position_history_players) == len(self.position_history)
            return self.position_history_players
        return (EMPTY,) * len(self.position_history)

    def _player_has_left_position(self, player: Player, board: Board) -> bool:
        return any(
            history_player == player and history_board == board
            for history_board, history_player in zip(
                self.position_history,
                self._history_players(),
                strict=True,
            )
        )

    def observation(self) -> list[float]:
        own = [
            1.0 if point == self.current_player else 0.0
            for point in self.board
        ]
        opponent = [
            1.0 if point == other_player(self.current_player) else 0.0
            for point in self.board
        ]
        return own + opponent

    def area_scores(self) -> tuple[float, float]:
        black_score = float(sum(1 for point in self.board if point == BLACK))
        white_score = float(sum(1 for point in self.board if point == WHITE)) + self.komi
        visited: set[int] = set()

        for point, color in enumerate(self.board):
            if color != EMPTY or point in visited:
                continue
            region, border_colors = _collect_empty_region(
                self.board,
                point,
                self.board_size,
            )
            visited.update(region)
            if WHITE not in border_colors:
                black_score += len(region)
            if BLACK not in border_colors:
                white_score += len(region)

        return black_score, white_score

    def winner(self) -> Player:
        black_score, white_score = self.area_scores()
        if black_score > white_score:
            return BLACK
        if white_score > black_score:
            return WHITE
        return EMPTY

    def rewards(self) -> dict[Player, float]:
        winner = self.winner()
        if winner == BLACK:
            return {BLACK: 1.0, WHITE: -1.0}
        if winner == WHITE:
            return {BLACK: -1.0, WHITE: 1.0}
        return {BLACK: 0.0, WHITE: 0.0}

    def result_for_player(self, player: Player) -> float:
        assert player in (BLACK, WHITE)
        return self.rewards()[player]

    def render_ascii(self) -> str:
        symbols = {
            BLACK: "X",
            WHITE: "O",
            EMPTY: ".",
        }
        rows = []
        for row in range(self.board_size):
            start = row * self.board_size
            stop = start + self.board_size
            rows.append(" ".join(symbols[point] for point in self.board[start:stop]))
        return "\n".join(rows)
