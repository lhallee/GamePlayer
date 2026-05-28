import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from game_player.reporting import write_training_report


class ReportingTest(unittest.TestCase):
    def test_write_training_report_creates_markdown_and_pngs(self):
        metrics = [
            {
                "iteration": 1.0,
                "loss": 3.0,
                "policy_loss": 2.0,
                "value_loss": 1.0,
                "self_play_samples": 10.0,
                "replay_size": 10.0,
                "validation_candidate_win_rate": 0.25,
                "validation_candidate_black_win_rate": 0.5,
                "validation_candidate_white_win_rate": 0.0,
                "validation_candidate_avg_score_margin": -4.0,
                "validation_avg_moves": 20.0,
                "validation_terminal_rate": 0.25,
            },
            {
                "iteration": 2.0,
                "loss": 2.0,
                "policy_loss": 1.5,
                "value_loss": 0.5,
                "self_play_samples": 12.0,
                "replay_size": 22.0,
                "validation_candidate_win_rate": 0.75,
                "validation_candidate_black_win_rate": 1.0,
                "validation_candidate_white_win_rate": 0.5,
                "validation_candidate_avg_score_margin": 8.0,
                "validation_avg_moves": 18.0,
                "validation_terminal_rate": 0.5,
            },
        ]
        with TemporaryDirectory() as tmp_dir:
            report_dir = Path(tmp_dir)
            write_training_report(
                metrics=metrics,
                report_dir=report_dir,
                run_config={"board_size": 5},
                title="Smoke Report",
            )
            self.assertTrue((report_dir / "report.md").exists())
            self.assertTrue((report_dir / "loss.png").exists())
            self.assertTrue((report_dir / "validation_win_rate.png").exists())
            self.assertTrue((report_dir / "validation_color_win_rate.png").exists())
            self.assertTrue((report_dir / "validation_score_margin.png").exists())
            self.assertTrue((report_dir / "validation_game_dynamics.png").exists())
            self.assertTrue((report_dir / "self_play.png").exists())


if __name__ == "__main__":
    unittest.main()
