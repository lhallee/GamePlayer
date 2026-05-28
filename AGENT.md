# Game Player Agent Notes

## Project Goal

Build a modular AlphaZero-style self-play framework for multiple games, starting with Go.
The project should stay lightweight, Python-first, and easy to extend.

## Non-Negotiables

- No supervised imitation-learning path for the core training loop.
- Training data comes from self-play, optionally guided by MCTS.
- Base inference should be direct policy argmax, not MCTS.
- Prefer pure PyTorch modules and Hugging Face-compatible model packaging.
- Keep game rules, simulation, models, training, evaluation, and frontends separate.
- Do not auto-commit or push.

## Current Design Direction

- `game_player/games/`: game interfaces plus per-game rule implementations.
- `game_player/az/`: AlphaZero primitives such as MCTS, replay, self-play, training, evaluation.
- `game_player/models/`: reusable policy-value models.
- `game_player/agents/`: random, argmax, and optional MCTS-backed agents.
- `apps/`: playable frontends such as Streamlit.
- `rules/`: Markdown rules files for human-readable game definitions.
- `tests/`: lightweight checks for rules, model shapes, MCTS, and smoke training.

## Go Case Study

- Board size defaults to 19x19, with tests allowed to use 5x5 or 9x9 for speed.
- Observation is from the current player's perspective:
  - own stones channel
  - opponent stones channel
- Primary policy output is board logits shaped `[board_size, board_size]`.
- Go still needs an internal pass action for termination and scoring. Expose this explicitly in action helpers rather than hiding it in board logits.
- Implement Tromp-Taylor captures, legal suicide with own-stone emptying,
  per-player repeated-position prevention, consecutive-pass termination, and
  area scoring with komi.
- Prefer Tromp-Taylor area scoring for automation; document rule assumptions.

## AlphaZero Notes

- Use a single policy-value network.
- MCTS uses the model policy as prior and value head as leaf evaluator.
- Self-play stores `(observation, improved_policy, outcome)` samples.
- Learner optimizes policy cross-entropy plus value MSE.
- Replay buffer should be FIFO and sample uniformly at this stage.
- Validation should run match batches against fixed baselines, including a random or random-weight opponent.
- Logs should be machine-readable JSONL where practical.
- Training should support repeated self-play/train/evaluate iterations, not just
  one-off smoke runs.
- Training runs should produce JSONL metrics, PNG plots at 300 dpi, and a
  Markdown report when a report directory is configured.

## Implementation Bias

- Start with an MLP policy-value model for the requested MOP:
  - flatten two board channels
  - hidden MLP torso
  - board policy head
  - value head
  - separate pass logit when the game supports pass
- Keep APIs generic enough for future card games, but do not overbuild hidden-information support until needed.
- Make scripts small and composable.
- Keep heavy training commands documented but do not run intensive training inline.
- Prefer checkpoint directories that are directly compatible with
  `huggingface_hub.PyTorchModelHubMixin`.

## Research Anchors

- AlphaGo Zero: pure self-play, combined policy-value net, no human games, no rollouts.
- AlphaZero: same general algorithm across chess, shogi, and Go with game rules only.
- OpenSpiel: actors generate MCTS self-play, learner trains from replay, evaluators track progress.
- Minigo: practical Go implementation pattern with bootstrap, self-play, train, validate.
- KataGo and ELF OpenGo: useful references for Go-specific acceleration, not the initial complexity target.
- Tromp-Taylor or AGA-style area scoring is the simplest fit for automated Go scoring.
