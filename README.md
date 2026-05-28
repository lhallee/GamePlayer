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

The default Go training model is a small convolutional policy-value network
with an optional ownership head. The MLP remains useful for smoke tests.

The current Go observation has two flattened board channels:

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

The app can play against a random legal agent, random-weight argmax MLP, or an
MLP checkpoint loaded from a local directory or Hugging Face Hub repo.

Run a tiny local training smoke:

```powershell
game-player-train-go --board-size 5 --self-play-games 2 --mcts-simulations 8 --train-steps 2
```

Run iterative self-play with JSONL metrics and Hugging Face-compatible
checkpoints:

```powershell
game-player-train-go --board-size 5 --iterations 5 --self-play-games 4 --mcts-simulations 16 --train-steps 8 --batch-size 16 --validation-games 20 --metrics-path runs/go5/metrics.jsonl --report-dir runs/go5/report --checkpoint-dir runs/go5/checkpoints
```

Each checkpoint directory can be loaded with `MLPPolicyValueNet.from_pretrained`
or `ConvPolicyValueNet.from_pretrained`, depending on `--model-type`, or pushed
with `model.push_to_hub(...)`.

When `--report-dir` is provided, training writes:

- `report.md`
- `loss.png`
- `validation_win_rate.png`
- `validation_color_win_rate.png`
- `validation_score_margin.png`
- `validation_game_dynamics.png`
- `self_play.png`

Evaluate a saved Go checkpoint against a random-weight model:

```powershell
game-player-eval-go --model runs/go5/checkpoints/iteration-0001 --opponent random-weights --games 100
```

## Training Recipes

Tiny local smoke run:

```powershell
game-player-train-go --board-size 5 --komi 2.5 --model-type mlp --hidden-size 128 --depth 2 --iterations 3 --self-play-games 2 --mcts-simulations 4 --train-steps 4 --batch-size 8 --max-moves 50 --validation-games 4 --validation-opponent random-agent --metrics-path runs/go5_smoke/metrics.jsonl --report-dir runs/go5_smoke/report --checkpoint-dir runs/go5_smoke/checkpoints
```

Current 9x9 Conv tactical bootstrap recipe:

```powershell
game-player-train-go --board-size 9 --komi 7.5 --model-type conv --channels 32 --blocks 3 --iterations 24 --self-play-games 8 --mcts-simulations 1 --mcts-evaluator tactical --value-target score --augment-symmetries --ownership-loss-weight 0.5 --train-steps 192 --batch-size 128 --max-moves 120 --validation-games 100 --validation-opponent random-agent --validation-max-moves 120 --target-win-rate 0.99 --seed 3 --metrics-path runs/go9_conv_tactical/metrics.jsonl --report-dir runs/go9_conv_tactical/report --checkpoint-dir runs/go9_conv_tactical/checkpoints --best-checkpoint-dir runs/go9_conv_tactical/best --run-name "Go 9x9 Conv Tactical Bootstrap"
```

This recipe trains with a rules-only tactical search teacher and validates
argmax-only network inference against uniform random legal play. It is a
bootstrap recipe, not pure AlphaGo Zero.

Current 19x19 continuation recipe:

```powershell
game-player-train-go --board-size 19 --komi 7.5 --model-type conv --channels 32 --blocks 3 --init-model runs/go19_conv_tactical_probe/checkpoints/iteration-0008 --iterations 20 --self-play-games 2 --mcts-simulations 1 --mcts-evaluator tactical --tactical-area-weight 0 --value-target score --augment-symmetries --ownership-loss-weight 0.5 --train-steps 512 --batch-size 128 --max-moves 400 --validation-games 40 --validation-interval 5 --validation-opponent random-agent --validation-max-moves 400 --target-win-rate 0.95 --seed 5 --metrics-path runs/go19_conv_tactical_continue/metrics.jsonl --report-dir runs/go19_conv_tactical_continue/report --checkpoint-dir runs/go19_conv_tactical_continue/checkpoints --best-checkpoint-dir runs/go19_conv_tactical_continue/best --run-name "Go 19x19 Conv Tactical Continue"
```

Workstation Docker version:

```bash
sudo docker run --rm --ipc=host \
  -v /home/ubuntu/GamePlayer/runs:/workspace/runs \
  gameplayer:dev \
  python -m game_player.scripts.train_go \
    --board-size 19 \
    --komi 7.5 \
    --model-type conv \
    --channels 32 \
    --blocks 3 \
    --init-model runs/go19_conv_tactical_probe/checkpoints/iteration-0008 \
    --iterations 20 \
    --self-play-games 2 \
    --mcts-simulations 1 \
    --mcts-evaluator tactical \
    --tactical-area-weight 0 \
    --value-target score \
    --augment-symmetries \
    --ownership-loss-weight 0.5 \
    --train-steps 512 \
    --batch-size 128 \
    --max-moves 400 \
    --validation-games 40 \
    --validation-interval 5 \
    --validation-opponent random-agent \
    --validation-max-moves 400 \
    --target-win-rate 0.95 \
    --seed 5 \
    --metrics-path runs/go19_conv_tactical_continue/metrics.jsonl \
    --report-dir runs/go19_conv_tactical_continue/report \
    --checkpoint-dir runs/go19_conv_tactical_continue/checkpoints \
    --best-checkpoint-dir runs/go19_conv_tactical_continue/best \
    --run-name "Go 19x19 Conv Tactical Continue"
```

Use `--validation-interval` to reduce validation cost on large boards while
still validating the final iteration.

Evaluate the current rules-only tactical teacher before distilling it:

```powershell
game-player-eval-tactical-go --board-size 19 --komi 7.5 --games 40 --max-moves 400 --tactical-area-weight 0
```

Verified workstation checkpoints:

- 9x9 Conv tactical bootstrap:
  `runs/go9_conv_tactical_99/best`, 97/100 seed-0 eval and 197/200 seed-11
  eval against uniform random legal play.
- 19x19 Conv atari tactical continuation:
  `runs/go19_conv_tactical_atari_continue/best`, 99/100 seed-0 eval against
  uniform random legal play with 49/50 as black and 50/50 as white.

These are tactical-bootstrap agents. They use rules-derived search targets and
score or ownership targets, not human game supervision.

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

See [docs/alphazero-muzero-literature.md](docs/alphazero-muzero-literature.md)
for implementation-focused notes on AlphaGo Zero, AlphaZero, MuZero, Minigo,
OpenSpiel, ELF OpenGo, and KataGo.

## Workstation Repro

See [docs/workstation-run.md](docs/workstation-run.md) for the SSH workstation
setup, Docker build/test commands, and verified 5x5 and 19x19 self-play runs
against fixed random-weight validation baselines.
