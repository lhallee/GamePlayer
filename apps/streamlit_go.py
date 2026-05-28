from __future__ import annotations

import streamlit as st

from game_player.agents.argmax_agent import ArgmaxPolicyAgent
from game_player.agents.random_agent import RandomAgent
from game_player.games.go import BLACK, EMPTY, WHITE, GoState
from game_player.models.mlp import MLPPolicyValueNet


def main() -> None:
    st.set_page_config(page_title="Game Player Go", layout="wide")
    st.title("Game Player Go")

    board_size = st.sidebar.selectbox("Board", [5, 9, 13, 19], index=1)
    agent_kind = st.sidebar.selectbox("Agent", ["Random", "Argmax MLP"], index=0)
    model_path = st.sidebar.text_input("Model path or Hub repo", value="")
    if st.sidebar.button("New Game"):
        st.session_state.go_state = GoState.new(board_size=board_size)

    if "go_state" not in st.session_state:
        st.session_state.go_state = GoState.new(board_size=board_size)

    state = st.session_state.go_state
    if state.board_size != board_size:
        state = GoState.new(board_size=board_size)
        st.session_state.go_state = state

    st.caption(_status_text(state))
    _draw_board(state, agent_kind, model_path)

    black_score, white_score = st.session_state.go_state.area_scores()
    st.write(
        {
            "black": black_score,
            "white": white_score,
            "moves until terminal": "done" if state.is_terminal() else "playing",
        }
    )


def _draw_board(state: GoState, agent_kind: str, model_path: str) -> None:
    legal_actions = set(state.legal_actions())
    for row in range(state.board_size):
        columns = st.columns(state.board_size, gap="small")
        for col, column in enumerate(columns):
            action = row * state.board_size + col
            label = _stone_label(state.board[action])
            disabled = state.is_terminal() or action not in legal_actions
            if column.button(
                label,
                key=f"point-{row}-{col}-{len(state.position_history)}",
                disabled=disabled,
                use_container_width=True,
            ):
                _play_human_action(action, agent_kind, model_path)
                st.rerun()

    if st.button("Pass", disabled=state.is_terminal()):
        _play_human_action(state.pass_action, agent_kind, model_path)
        st.rerun()


def _play_human_action(action: int, agent_kind: str, model_path: str) -> None:
    state = st.session_state.go_state
    if state.current_player != BLACK:
        return

    state = state.apply_action(action)
    if not state.is_terminal():
        agent = _build_agent(agent_kind, state.board_size, model_path)
        agent_action = agent.select_action(state)
        state = state.apply_action(agent_action)
    st.session_state.go_state = state


def _build_agent(agent_kind: str, board_size: int, model_path: str):
    if agent_kind == "Random":
        return RandomAgent()
    if model_path:
        return _load_argmax_agent(model_path, board_size)
    return _build_random_weight_argmax_agent(board_size)


@st.cache_resource
def _build_random_weight_argmax_agent(board_size: int):
    model = MLPPolicyValueNet(board_size=board_size, hidden_size=128, depth=2)
    return ArgmaxPolicyAgent(model)


@st.cache_resource
def _load_argmax_agent(model_path: str, board_size: int):
    model = MLPPolicyValueNet.from_pretrained(model_path)
    assert model.board_size == board_size
    return ArgmaxPolicyAgent(model)


def _stone_label(stone: int) -> str:
    if stone == BLACK:
        return "X"
    if stone == WHITE:
        return "O"
    if stone == EMPTY:
        return "."
    raise ValueError(f"Unknown stone value: {stone}")


def _status_text(state: GoState) -> str:
    if state.is_terminal():
        winner = state.winner()
        if winner == BLACK:
            return "Black wins"
        if winner == WHITE:
            return "White wins"
        return "Tie"
    if state.current_player == BLACK:
        return "Black to move"
    return "White to move"


if __name__ == "__main__":
    main()
