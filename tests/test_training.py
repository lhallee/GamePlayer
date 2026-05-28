import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import torch

    from game_player.scripts.train_go import run_training_cli
    from game_player.az.training import TrainingConfig, run_training_iterations
    from game_player.models.mlp import MLPPolicyValueNet
except ModuleNotFoundError:
    torch = None
    run_training_cli = None
    TrainingConfig = None
    run_training_iterations = None
    MLPPolicyValueNet = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrainingTest(unittest.TestCase):
    def test_iterative_training_returns_metrics_per_iteration(self):
        model = MLPPolicyValueNet(board_size=3, hidden_size=16, depth=1)
        config = TrainingConfig(
            board_size=3,
            komi=0.5,
            iterations=2,
            self_play_games=1,
            mcts_simulations=2,
            batch_size=2,
            train_steps=1,
            max_moves=4,
            seed=0,
        )
        metrics = run_training_iterations(
            model=model,
            config=config,
        )
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[0]["iteration"], 1.0)
        self.assertEqual(metrics[1]["iteration"], 2.0)
        self.assertGreater(metrics[1]["replay_size"], metrics[0]["replay_size"])
        self.assertEqual(metrics[1]["train_steps_completed"], 1.0)

    def test_cli_training_writes_metrics_and_checkpoint(self):
        model = MLPPolicyValueNet(board_size=3, hidden_size=16, depth=1)
        baseline = MLPPolicyValueNet(board_size=3, hidden_size=16, depth=1)
        config = TrainingConfig(
            board_size=3,
            komi=0.5,
            iterations=1,
            self_play_games=1,
            mcts_simulations=2,
            batch_size=2,
            train_steps=1,
            max_moves=4,
            seed=0,
        )

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            metrics = run_training_cli(
                model=model,
                config=config,
                baseline_model=baseline,
                validation_games=1,
                validation_opponent="random-agent",
                validation_max_moves=4,
            target_win_rate=None,
            checkpoint_dir=root / "checkpoints",
            best_checkpoint_dir=root / "best",
            save_every=1,
            metrics_path=root / "metrics.jsonl",
            report_dir=root / "report",
            run_config={"board_size": 3},
            run_name="Training Smoke",
            device="cpu",
            mcts_evaluator="rollout",
            rollout_games=1,
            rollout_max_moves=4,
        )
            self.assertEqual(len(metrics), 1)
            self.assertTrue((root / "metrics.jsonl").exists())
            self.assertTrue((root / "checkpoints" / "iteration-0001").exists())
            self.assertTrue(
                (root / "checkpoints" / "iteration-0001" / "config.json").exists()
            )
            self.assertTrue((root / "best" / "config.json").exists())
            self.assertTrue((root / "report" / "report.md").exists())
            self.assertTrue((root / "report" / "loss.png").exists())


if __name__ == "__main__":
    unittest.main()
