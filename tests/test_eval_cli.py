import unittest
from tempfile import TemporaryDirectory

try:
    from game_player.hub import save_mlp_model
    from game_player.models.mlp import MLPPolicyValueNet
    from game_player.scripts.eval_go import build_parser
except ModuleNotFoundError:
    save_mlp_model = None
    MLPPolicyValueNet = None
    build_parser = None


@unittest.skipIf(MLPPolicyValueNet is None, "PyTorch is not installed")
class EvalCliTest(unittest.TestCase):
    def test_parser_accepts_saved_model_path(self):
        with TemporaryDirectory() as tmp_dir:
            model = MLPPolicyValueNet(board_size=3, hidden_size=16, depth=1)
            save_mlp_model(model, tmp_dir)
            args = build_parser().parse_args(
                [
                    "--model",
                    tmp_dir,
                    "--opponent",
                    "random-weights",
                    "--games",
                    "2",
                    "--komi",
                    "2.5",
                ]
            )
            self.assertEqual(args.model, tmp_dir)
            self.assertEqual(args.opponent, "random-weights")
            self.assertEqual(args.games, 2)
            self.assertEqual(args.komi, 2.5)


if __name__ == "__main__":
    unittest.main()
