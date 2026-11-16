# Contributing

Focused fixes and clear explanations are welcome. This is a learning project;
readability and understanding the math matter more than adding abstractions.

## Where changes belong

- `llm/`: the attention layers, Transformer blocks, and current model.
- `training/`: the streaming loader, optimizer loop, validation, and run outputs.
- `learning/`: isolated exercises and explicitly labeled historical drafts.
- `docs/`: setup, input-format, and training documentation.

## Before proposing a change

1. Keep the change focused on one concept or problem.
2. Preserve the author's explicit coding style where practical.
3. For model or training changes, explain the behavior and show a small check.
4. Report the command, environment, expected behavior, and observed result.
5. Keep local datasets, acquisition/cleaning tools, checkpoints, and credentials
   outside the published code tree.

A quick syntax check is:

```bash
python -m compileall -q llm training learning
```

Run a training check only with your own local corpus and a separate output folder.
Do not present an early training-loss value as a model-quality benchmark. When
reporting an evaluation, identify the dataset, metric, run settings, and limits.

## CPU attention regression checks

With PyTorch and tiktoken installed, run from the repository root:

```bash
python -m unittest discover -s tests -v
```

The four attention tests use tiny CPU tensors (8 input features, 12 output
features, three heads, and at most six tokens). They check finite outputs at
supported sequence lengths, batch independence, unchanged prefix outputs when
future tokens change, and zero gradients from a prefix-only loss to future
inputs. No corpus, tokenizer download, checkpoint, GPU, or training run is needed.

Validated with Python 3.13 and PyTorch 2.12.1+cu130 using CPU tensors only. These
checks verify the attention component's causal behavior with dropout disabled;
they do not measure language-model quality or validate the full training pipeline.

## Streaming dataset regression checks

The same discovery command also runs five streaming tests using temporary UTF-8
text and JSONL indexes. They verify byte offsets, shifted targets across document
boundaries, dropped short remainders, the pre-shuffle token cap, repeatable epochs,
real PyTorch DataLoader batching, invalid ranges, and the single-worker guard.

`tiktoken` must be importable, but its encoding lookup is replaced by a local
character tokenizer in these tests. No vocabulary or corpus is downloaded. These
checks validate loader mechanics, not GPT-2 tokenization or training quality.
