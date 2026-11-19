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

The same discovery command also runs seven streaming tests using temporary UTF-8
text and JSONL indexes. They verify byte offsets, shifted targets across document
boundaries, dropped short remainders, the pre-shuffle token cap, repeatable epochs,
real PyTorch DataLoader batching, invalid ranges, malformed index records with
line-number errors, and the single-worker guard.

`tiktoken` must be importable, but its encoding lookup is replaced by a local
character tokenizer in these tests. No vocabulary or corpus is downloaded. These
checks validate loader mechanics, not GPT-2 tokenization or training quality.

## Model integration regression checks

Discovery also runs three tests through the full 12-block `Model`, using eight
features, two attention heads, a vocabulary of 17, and at most six positions.
They check causal prefix logits, finite cross-entropy gradients reaching every
parameter, one synthetic SGD update, and an in-memory state-dict save/load round
trip that preserves evaluation logits exactly. The gradient check uses training
mode; output comparisons use evaluation mode to disable dropout.

Inputs are synthetic floating-point embeddings. These checks do not exercise the
training script's token/position embeddings, real tokenization, optimizer-state
resume, or a corpus. The single tiny CPU update is a wiring check, not evidence
of learning quality. Run with `OMP_NUM_THREADS=1` before the discovery command to
limit CPU thread overhead on small tensors.

## Automated CPU checks

The `CPU regression tests` GitHub Actions workflow runs the same discovery suite
on pushes to `main`, pull requests, and manual dispatch. It uses Ubuntu 24.04,
Python 3.13, CPU-only PyTorch 2.12.1, and tiktoken 0.14.0. Actions are pinned to
commit revisions, the job has read-only repository permissions, and its timeout
is ten minutes. No secrets, corpus, model checkpoint, or GPU are required.

To reproduce the direct dependencies in a fresh Python 3.13 virtual environment:

```bash
python -m pip install torch==2.12.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install tiktoken==0.14.0
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Transitive dependencies are resolved by pip rather than locked. The general
`requirements.txt` remains the dependency list for training and plotting; this
workflow installs only what the CPU tests need. Passing CI verifies the checks
described above, not end-to-end training or language-model quality.
