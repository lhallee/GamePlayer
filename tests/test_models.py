import unittest

try:
    import torch

    from game_player.models.conv import ConvPolicyValueNet
    from game_player.models.mlp import MLPPolicyValueNet
except ModuleNotFoundError:
    torch = None
    ConvPolicyValueNet = None
    MLPPolicyValueNet = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MLPModelTest(unittest.TestCase):
    def test_model_output_shapes(self):
        model = MLPPolicyValueNet(board_size=3, hidden_size=16, depth=2)
        observation = torch.zeros(2, 18)
        output = model(observation)
        self.assertEqual(tuple(output.policy_logits.shape), (2, 10))
        self.assertEqual(tuple(output.board_logits.shape), (2, 3, 3))
        self.assertEqual(tuple(output.value.shape), (2,))
        self.assertIsNone(output.ownership)

    def test_conv_model_output_shapes(self):
        model = ConvPolicyValueNet(board_size=3, channels=8, blocks=1)
        observation = torch.zeros(2, 18)
        output = model(observation)
        self.assertEqual(tuple(output.policy_logits.shape), (2, 10))
        self.assertEqual(tuple(output.board_logits.shape), (2, 3, 3))
        self.assertEqual(tuple(output.value.shape), (2,))
        self.assertIsNotNone(output.ownership)
        assert output.ownership is not None
        self.assertEqual(tuple(output.ownership.shape), (2, 3, 3))


if __name__ == "__main__":
    unittest.main()
