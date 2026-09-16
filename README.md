# TinyGPT

TinyGPT is an educational implementation of a GPT-style decoder-only language
model. The project is intended to make the mechanics of language-model
pretraining explicit and easy to inspect.

The long-term project will cover tokenizer training, language-model dataset
preparation, causal attention, a Transformer implementation, next-token
pretraining, text generation, and evaluation. The model will be built from
basic PyTorch primitives rather than a pretrained GPT implementation.

The repository currently contains Phase 1 infrastructure, the Phase 2
text-to-token pipeline, and the Phase 3 language-model dataset pipeline.
Model training and the GPT implementation have not been implemented.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
```

## Phase 2 tokenizer

Check access to TinyStories:

```bash
python scripts/download_data.py
```

Train the canonical tokenizer:

```bash
python scripts/train_tokenizer.py
```

For a limited development run:

```bash
python scripts/train_tokenizer.py --max-samples 10000
```

The canonical tokenizer is saved to
`artifacts/tokenizer/tokenizer.json`. Model training belongs to later phases.

## Phase 3 tokenized dataset

Prepare the TinyStories token streams using the existing tokenizer:

```bash
python scripts/prepare_data.py
```

This command reads the TinyStories dataset from the local HuggingFace cache
created during Phase 2 and does not download the dataset again. If the cache is
missing, run the Phase 2 data access or tokenizer preparation first.

For a small development run:

```bash
python scripts/prepare_data.py \
  --max-train-samples 1000 \
  --max-val-samples 100
```

The preparation pipeline is:

```text
TinyStories text
    -> tokenizer
    -> one EOS token per story
    -> contiguous uint16 token stream
    -> memory-mapped GPTDataset
```

The canonical output files are `data/train.bin` and `data/val.bin`. Each file
contains raw `uint16` token IDs without a header. `GPTDataset` reads the stream
with `numpy.memmap` and returns overlapping next-token pairs as `torch.long`
tensors:

```python
x = tokens[start : start + seq_len]
y = tokens[start + 1 : start + seq_len + 1]
```

Stories are separated with EOS and are not padded, truncated, or shuffled.
The GPT model, training loop, and training dataloader policy are still not
implemented.
