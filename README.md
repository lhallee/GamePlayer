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
or pushed with `model.push_to_hub(...)`.

When `--report-dir` is provided, training writes:

- `report.md`
- `loss.png`
- `validation_win_rate.png`
- `self_play.png`

Evaluate a saved Go checkpoint against a random-weight model:

```powershell
game-player-eval-go --model runs/go5/checkpoints/iteration-0001 --opponent random-weights --games 100
```

## Training Recipes

All current recipes use the `MLPPolicyValueNet` model:

- observation: current-player stones plus opponent stones
- policy head: board logits plus pass logit
- value head: scalar win/loss estimate
- self-play target: AlphaZero-style MCTS visit policy

Small 5x5 smoke run:

```powershell
game-player-train-go --board-size 5 --komi 2.5 --hidden-size 128 --depth 2 --iterations 5 --self-play-games 4 --mcts-simulations 8 --train-steps 16 --batch-size 32 --max-moves 60 --validation-games 40 --validation-opponent random-weights --validation-max-moves 80 --target-win-rate 0.9 --metrics-path runs/go5_mlp/metrics.jsonl --report-dir runs/go5_mlp/report --checkpoint-dir runs/go5_mlp/checkpoints --best-checkpoint-dir runs/go5_mlp/best
```

Verified 19x19 MLP run:

```powershell
game-player-train-go --board-size 19 --komi 7.5 --hidden-size 256 --depth 2 --iterations 10 --self-play-games 2 --mcts-simulations 4 --train-steps 8 --batch-size 16 --max-moves 120 --validation-games 40 --validation-opponent random-weights --validation-max-moves 160 --target-win-rate 0.9 --metrics-path runs/go19_mlp/metrics.jsonl --report-dir runs/go19_mlp/report --checkpoint-dir runs/go19_mlp/checkpoints --best-checkpoint-dir runs/go19_mlp/best --run-name "Go 19x19 MLP AlphaZero"
```

Workstation Docker version:

```bash
sudo docker run --rm --ipc=host \
  -v /home/ubuntu/GamePlayer/runs:/workspace/runs \
  gameplayer:dev \
  python -m game_player.scripts.train_go \
    --board-size 19 \
    --komi 7.5 \
    --hidden-size 256 \
    --depth 2 \
    --iterations 10 \
    --self-play-games 2 \
    --mcts-simulations 4 \
    --train-steps 8 \
    --batch-size 16 \
    --max-moves 120 \
    --validation-games 40 \
    --validation-opponent random-weights \
    --validation-max-moves 160 \
    --target-win-rate 0.9 \
    --metrics-path runs/go19_mlp/metrics.jsonl \
    --report-dir runs/go19_mlp/report \
    --checkpoint-dir runs/go19_mlp/checkpoints \
    --best-checkpoint-dir runs/go19_mlp/best \
    --run-name "Go 19x19 MLP AlphaZero"
```

On the workstation this recipe reached the target at iteration 6 and the best
checkpoint scored 100/100 against the fixed random-weight validation baseline.

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

## Workstation Repro

See [docs/workstation-run.md](docs/workstation-run.md) for the SSH workstation
setup, Docker build/test commands, and verified 5x5 and 19x19 self-play runs
against fixed random-weight validation baselines.
