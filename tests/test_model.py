"""CPU integration checks for the model stack, without embeddings or a corpus."""
from io import BytesIO
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.model import Model


class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = Model(
            d_in=8, d_out=8, d_head=4, context_length=6, vocab_size=17,
        ).cpu()

    def test_logits_and_causal_prefix_at_model_level(self):
        self.model.eval()
        inputs = torch.randn(2, 6, 8)
        changed = inputs.clone()
        changed[:, 3:] = torch.randn(2, 3, 8) * 10
        with torch.no_grad():
            logits = self.model(inputs)
            changed_logits = self.model(changed)
            prefix_logits = self.model(inputs[:, :3])
        self.assertEqual(logits.shape, (2, 6, 17))
        self.assertTrue(torch.isfinite(logits).all().item())
        torch.testing.assert_close(logits[:, :3], changed_logits[:, :3], rtol=0, atol=0)
        torch.testing.assert_close(logits[:, :3], prefix_logits, rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(logits[:, 3:], changed_logits[:, 3:]))

    def test_cross_entropy_reaches_every_parameter_and_updates_weights(self):
        self.model.train()
        inputs = torch.randn(2, 6, 8)
        targets = torch.randint(0, 17, (2, 6))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        before = self.model.linear.weight.detach().clone()
        optimizer.zero_grad()
        loss = F.cross_entropy(self.model(inputs).flatten(0, 1), targets.flatten())
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.linear.weight))
        for parameter in self.model.parameters():
            self.assertTrue(torch.isfinite(parameter).all().item())

    def test_state_dict_round_trip_preserves_eval_logits(self):
        self.model.eval()
        inputs = torch.randn(2, 4, 8)
        buffer = BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = Model(8, 8, 4, 6, 17).cpu().eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(self.model(inputs), restored(inputs), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
