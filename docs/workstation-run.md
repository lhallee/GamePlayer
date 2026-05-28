# Workstation Run Notes

Workstation:

```text
ssh -i C:/Users/Logan/.ssh/AP_bio_logan.txt ubuntu@192.222.51.17
```

Remote repo path:

```text
/home/ubuntu/GamePlayer
```

The repo was cloned from:

```text
https://github.com/lhallee/GamePlayer.git
```

The local working tree was then synced over the clone because the current
implementation changes were not committed or pushed.

## Docker

Build:

```bash
cd /home/ubuntu/GamePlayer
sudo docker build -t gameplayer:dev .
```

Test:

```bash
sudo docker run --rm --ipc=host gameplayer:dev
```

Observed result:

```text
Ran 16 tests in 1.727s
OK
```

## Cheap Go Training Run

Command:

```bash
sudo docker run --rm --ipc=host \
  -v /home/ubuntu/GamePlayer/runs:/workspace/runs \
  gameplayer:dev \
  python -m game_player.scripts.train_go \
    --board-size 5 \
    --komi 2.5 \
    --hidden-size 128 \
    --depth 2 \
    --iterations 20 \
    --self-play-games 4 \
    --mcts-simulations 8 \
    --train-steps 16 \
    --batch-size 32 \
    --max-moves 60 \
    --validation-games 40 \
    --validation-opponent random-weights \
    --validation-max-moves 80 \
    --target-win-rate 0.9 \
    --metrics-path runs/go5_rw/metrics.jsonl \
    --checkpoint-dir runs/go5_rw/checkpoints \
    --best-checkpoint-dir runs/go5_rw/best
```

Observed result:

```json
{
  "iteration": 1.0,
  "loss": 3.961766555905342,
  "policy_loss": 3.2150015085935593,
  "replay_size": 102.0,
  "self_play_samples": 102.0,
  "target_reached": 1.0,
  "train_steps_completed": 16.0,
  "validation_candidate_black_games": 20.0,
  "validation_candidate_white_games": 20.0,
  "validation_candidate_win_rate": 1.0,
  "validation_candidate_wins": 40.0,
  "validation_games": 40.0,
  "value_loss": 0.7467650342732668
}
```

Best checkpoint:

```text
/home/ubuntu/GamePlayer/runs/go5_rw/best
```

Files:

```text
README.md
config.json
model.safetensors
```

## Independent Eval

Command:

```bash
sudo docker run --rm --ipc=host \
  -v /home/ubuntu/GamePlayer/runs:/workspace/runs \
  gameplayer:dev \
  python -m game_player.scripts.eval_go \
    --model runs/go5_rw/best \
    --opponent random-weights \
    --games 100 \
    --max-moves 80 \
    --komi 2.5 \
    --seed 0
```

Observed result:

```json
{
  "validation_candidate_black_games": 50.0,
  "validation_candidate_white_games": 50.0,
  "validation_candidate_win_rate": 1.0,
  "validation_candidate_wins": 100.0,
  "validation_games": 100.0
}
```

## Verified 19x19 Go Training Run

Command:

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
    --seed 0 \
    --metrics-path runs/go19_mlp_20260528_probe/metrics.jsonl \
    --report-dir runs/go19_mlp_20260528_probe/report \
    --checkpoint-dir runs/go19_mlp_20260528_probe/checkpoints \
    --best-checkpoint-dir runs/go19_mlp_20260528_probe/best \
    --run-name "Go 19x19 MLP AlphaZero Probe"
```

Observed result:

```json
{
  "iteration": 6.0,
  "loss": 5.904493868350983,
  "policy_loss": 5.6919360756874084,
  "replay_size": 1440.0,
  "self_play_samples": 240.0,
  "target_reached": 1.0,
  "train_steps_completed": 8.0,
  "validation_candidate_black_games": 20.0,
  "validation_candidate_white_games": 20.0,
  "validation_candidate_win_rate": 1.0,
  "validation_candidate_wins": 40.0,
  "validation_games": 40.0,
  "value_loss": 0.2125578671693802
}
```

Best checkpoint:

```text
/home/ubuntu/GamePlayer/runs/go19_mlp_20260528_probe/best
```

Report files:

```text
loss.png 91627
report.md 1913
self_play.png 104802
validation_win_rate.png 84291
```

Independent fixed-baseline eval:

```bash
sudo docker run --rm --ipc=host \
  -v /home/ubuntu/GamePlayer/runs:/workspace/runs \
  gameplayer:dev \
  python -m game_player.scripts.eval_go \
    --model runs/go19_mlp_20260528_probe/best \
    --opponent random-weights \
    --games 100 \
    --max-moves 160 \
    --komi 7.5 \
    --seed 0
```

Observed result:

```json
{
  "validation_candidate_black_games": 50.0,
  "validation_candidate_white_games": 50.0,
  "validation_candidate_win_rate": 1.0,
  "validation_candidate_wins": 100.0,
  "validation_games": 100.0
}
```

Random legal-move agent spot check:

```json
{
  "validation_candidate_black_games": 50.0,
  "validation_candidate_white_games": 50.0,
  "validation_candidate_win_rate": 0.47,
  "validation_candidate_wins": 47.0,
  "validation_games": 100.0
}
```

## Current Limitation

These checkpoints are high-performing against the fixed random-weight baseline
used by the inexpensive training recipes. The 5x5 checkpoint did not generalize
to a different fresh random-weight seed in a 200-game spot check, and the 19x19
checkpoint scored 47/100 against a uniform random legal-move agent. Treat these
runs as proof that the self-play training/eval/checkpoint/report loop works,
not as strong general Go agents.
