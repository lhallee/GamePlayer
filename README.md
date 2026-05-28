# Game Player

Game Player is a modular AlphaZero-style self-play playground for games. The
first case study is Go.

The repository is intentionally split by responsibility:

- `rules/`: human-readable game rules.
- `game_player/games/`: executable game states and legal move logic.
- `game_player/models/`: PyTorch policy-value models.
- `game_player/az/`: AlphaZero self-play, MCTS, replay, training, and eval.
- `game_player/agents/`: inference-time agents.
- `apps/`: playable frontends.
- `tests/`: lightweight correctness and smoke checks.

## Design Notes

The initial Go implementation follows the AlphaZero shape:

1. A single policy-value network predicts move priors and position value.
2. MCTS uses the policy head as a prior and the value head as a leaf evaluator.
3. Self-play stores `(observation, improved_policy, outcome)` samples.
4. Training minimizes policy cross-entropy plus value MSE.
5. Direct inference uses argmax over legal policy logits, without MCTS.

The default model is a small MLP over two flattened board channels:

- current player's stones
- opponent stones

The primary board policy is `[board_size, board_size]`. Go also needs a pass
action for termination, so the model exposes a separate pass logit when
`include_pass=True`.

## Quickstart

Install into a Python environment with PyTorch:

```powershell
py -3 -m pip install -e ".[dev]"
```

Run tests:

```powershell
py -3 -m unittest discover -s tests
```

Launch the Go app:

```powershell
streamlit run apps/streamlit_go.py
```

Run a tiny local training smoke:

```powershell
game-player-train-go --board-size 5 --self-play-games 2 --mcts-simulations 8 --train-steps 2
```

## Research Anchors

- DeepMind's AlphaGo Zero removed human games and trained from self-play using a
  combined policy-value network:
  https://deepmind.google/blog/alphago-zero-starting-from-scratch/
- AlphaZero generalized that approach across chess, shogi, and Go using game
  rules only:
  https://arxiv.org/abs/1712.01815
- OpenSpiel's AlphaZero implementation separates actors, learner, evaluator,
  replay, checkpoints, and analysis logs:
  https://openspiel.readthedocs.io/en/latest/alpha_zero.html
- Minigo is the most practical public Go reference for bootstrap, self-play,
  train, and validation workflow:
  https://github.com/tensorflow/minigo
- KataGo and ELF OpenGo are important later references for making Go training
  compute-efficient, but they add complexity beyond this first scaffold:
  https://arxiv.org/abs/1902.10565
