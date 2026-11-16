"""Streaming and batching tests using synthetic UTF-8 files and a local tokenizer."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.dataset import GPTDataset, create_dataloader_from_file


class CharacterTokenizer:
    eot_token = 1000

    def encode(self, text):
        return [ord(character) for character in text]


def samples(dataset):
    return [(tuple(x.tolist()), tuple(y.tolist())) for x, y in dataset]


class StreamingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.text = Path(self.folder.name) / "train.txt"
        self.index = self.text.with_suffix(".documents.jsonl")
        tokenizer = patch("training.dataset.tiktoken.get_encoding", return_value=CharacterTokenizer())
        tokenizer.start()
        self.addCleanup(tokenizer.stop)

    def write_documents(self, texts):
        contents = bytearray()
        entries = []
        for text in texts:
            start = len(contents)
            contents.extend(text.encode("utf-8"))
            entries.append({"byte_start": start, "byte_end": len(contents)})
        self.text.write_bytes(contents)
        self.index.write_text("".join(json.dumps(entry) + "\n" for entry in entries))

    def test_utf8_offsets_shifted_targets_and_document_boundary(self):
        self.write_documents(["éab", "cdef"])
        dataset = GPTDataset(self.text, context_length=3, shuffle_buffer_size=0)
        result = samples(dataset)
        self.assertEqual(result, [
            ((233, 97, 98), (97, 98, 1000)),
            ((1000, 99, 100), (99, 100, 101)),
        ])
        # The short trailing remainder is dropped, not padded or duplicated.
        self.assertEqual(len(result), 2)

    def test_cap_is_applied_before_shuffle_and_epoch_is_repeatable(self):
        self.write_documents(["abcdefghijklmnopqrstuvwxyz"])
        ordered = samples(GPTDataset(self.text, context_length=2, shuffle_buffer_size=0))
        dataset = GPTDataset(self.text, context_length=2, max_tokens=11,
                             shuffle_buffer_size=3, seed=42)
        first = samples(dataset)
        self.assertEqual(first, samples(dataset))
        self.assertCountEqual(first, ordered[:5])
        dataset.set_epoch(1)
        second = samples(dataset)
        self.assertEqual(second, samples(dataset))
        self.assertCountEqual(second, ordered[:5])
        self.assertNotEqual(first, second)

    def test_real_dataloader_batches_and_drop_last(self):
        self.write_documents(["abcdefghij"])
        for drop_last, sizes in [(False, [2, 2, 1]), (True, [2, 2])]:
            with self.subTest(drop_last=drop_last):
                batches = list(create_dataloader_from_file(
                    self.text, batch_size=2, context_length=2,
                    shuffle=False, drop_last=drop_last))
                self.assertEqual([x.shape[0] for x, _ in batches], sizes)
                for x, y in batches:
                    self.assertEqual(x.shape, y.shape)
                    self.assertEqual(x.shape[1], 2)
                    self.assertEqual(x.dtype, torch.long)
                    self.assertEqual(y.dtype, torch.long)

    def test_invalid_document_ranges(self):
        self.write_documents(["abcdef"])
        cases = [
            [{"byte_start": -1, "byte_end": 2}],
            [{"byte_start": 2, "byte_end": 2}],
            [{"byte_start": 0, "byte_end": 7}],
            [{"byte_start": 0, "byte_end": 4}, {"byte_start": 3, "byte_end": 6}],
        ]
        for entries in cases:
            self.index.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
            with self.subTest(entries=entries), self.assertRaisesRegex(ValueError, "Invalid document offsets"):
                list(GPTDataset(self.text, context_length=2))

    def test_worker_guard(self):
        self.write_documents(["abcdef"])
        with patch("training.dataset.get_worker_info", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "num_workers=0"):
                list(GPTDataset(self.text, context_length=2))


if __name__ == "__main__":
    unittest.main()
