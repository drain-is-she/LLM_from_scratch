"""Small CPU checks of causal attention; no corpus or checkpoints required."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.attention import MultiHeadAttention


class CausalAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.layer = MultiHeadAttention(
            d_in=8, d_out=12, context_length=6, dropout=0.0, num_heads=3,
            qkv_bias=True,
        ).cpu().eval()

    def test_shapes_and_finite_outputs_at_supported_lengths(self):
        for length in [1, 3, 6]:
            with self.subTest(length=length), torch.no_grad():
                output = self.layer(torch.randn(2, length, 8, device="cpu"))
                self.assertEqual(output.shape, (2, length, 12))
                self.assertTrue(torch.isfinite(output).all().item())

    def test_future_tokens_cannot_change_prefix(self):
        inputs = torch.randn(2, 6, 8, device="cpu")
        changed = inputs.clone()
        changed[:, 3:] = torch.randn(2, 3, 8, device="cpu") * 10
        with torch.no_grad():
            original_output = self.layer(inputs)
            changed_output = self.layer(changed)
            prefix_only = self.layer(inputs[:, :3])
        torch.testing.assert_close(original_output[:, :3], changed_output[:, :3], rtol=0, atol=0)
        torch.testing.assert_close(original_output[:, :3], prefix_only, rtol=1e-5, atol=1e-6)
        # Guard against a constant-output implementation passing the prefix test.
        self.assertFalse(torch.allclose(original_output[:, 3:], changed_output[:, 3:]))

    def test_prefix_loss_has_no_gradient_to_future_inputs(self):
        inputs = torch.randn(2, 6, 8, device="cpu", requires_grad=True)
        self.layer(inputs)[:, :3].square().sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all().item())
        self.assertGreater(inputs.grad[:, :3].abs().sum().item(), 0)
        self.assertEqual(torch.count_nonzero(inputs.grad[:, 3:]).item(), 0)
        for name, parameter in self.layer.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_batch_members_are_independent(self):
        inputs = torch.randn(2, 6, 8, device="cpu")
        with torch.no_grad():
            batched = self.layer(inputs)
            separate = torch.cat([self.layer(inputs[i:i + 1]) for i in range(2)])
        torch.testing.assert_close(batched, separate, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
