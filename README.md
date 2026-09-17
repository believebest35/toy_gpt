# TinyGPT

TinyGPT is an educational implementation of a GPT-style decoder-only language
model. The project is intended to make the mechanics of language-model
pretraining explicit and easy to inspect.

The long-term project will cover tokenizer training, language-model dataset
preparation, causal attention, a Transformer implementation, next-token
pretraining, text generation, and evaluation. The model will be built from
basic PyTorch primitives rather than a pretrained GPT implementation.

The repository currently contains Phase 1 infrastructure, the Phase 2
text-to-token pipeline, the Phase 3 language-model dataset pipeline, the Phase
4 attention components, the Phase 5 Transformer block, the Phase 6 full
TinyGPT model, and Phase 7 training/generation infrastructure.

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
`artifacts/tokenizer/tokenizer.json`. The training scripts use this local
artifact and do not download it again.

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
Training scripts sample windows from these memory-mapped streams.

## Phase 4 model components

Phase 4 adds learned token embeddings, learned absolute positional embeddings,
and explicit causal multi-head self-attention.

The attention implementation is intentionally educational. Its forward pass
exposes the main operations directly:

```text
Q, K, V
    -> scaled QK^T
    -> causal mask
    -> softmax
    -> weighted V
    -> output projection
```

For an input with shape `[B, T, C]`, attention reshapes the projections into
multiple heads, prevents each position from attending to future positions, and
returns `[B, T, C]`.

## Phase 5 Transformer block

Phase 5 adds a GPT-style GELU feed-forward MLP and one Pre-LayerNorm
Transformer block. The MLP expands the hidden dimension and projects it back:

```text
n_embd -> mlp_ratio * n_embd -> GELU -> n_embd
```

The block applies LayerNorm and residual connections in this order:

```text
x = x + Attention(LN(x))
x = x + MLP(LN(x))
```

The block is reusable inside the complete model added in Phase 6.

## Phase 6 complete TinyGPT model

Phase 6 stacks `TransformerBlock` modules into a decoder-only GPT model with a
final LayerNorm and a biasless language-model head. The head shares the same
Parameter as the token embedding, and the forward pass returns logits with
shape `[B, T, vocab_size]`. Passing shifted dataset targets additionally
returns a token-level cross-entropy loss; the model does not shift targets a
second time.

The model includes GPT-style weight initialization and supports ordinary
forward and backward passes. Training infrastructure is added in Phase 7;
mixed precision, schedulers, distributed training, KV caching, and
FlashAttention remain intentionally deferred.

## Phase 7 one-batch overfit sanity check

Phase 7 verifies that the complete GPT can learn through gradient descent. The
sanity-check script loads `configs/debug.yaml` and the local `data/train.bin`,
builds one fixed batch, and trains on that same batch with AdamW:

```bash
python scripts/overfit_batch.py
```

The script prints the device, batch shape, initial loss, periodic losses, and
final loss. It does not download data or implement the full pretraining loop.
The canonical `data/train.bin` uses the 8192-token tokenizer while the debug
model has a 256-token vocabulary, so this debug-only script maps the selected
token IDs modulo 256 before training. The dataset files themselves are not
modified.

## Small real-training experiment

To train on continuously sampled, different batches from the canonical
`GPTDataset` while monitoring validation loss, run:

```bash
python scripts/train_small.py
```

This uses `configs/tiny.yaml`, the local `data/train.bin` and `data/val.bin`,
AdamW, and 3000 optimization steps by default. Validation loss is measured on
20 fixed random validation batches sampled without replacement from the full
validation stream every 100 steps. This script remains a lightweight
experiment without checkpointing or resume support.

## Simple generation demo

The model can now generate text with repeated full forward passes and no KV
cache. To retrain the small model and inspect the same prompt every 500 steps
through step 3000:

```bash
python scripts/train_generate.py \
  --steps 3000 \
  --eval-interval 500 \
  --eval-batches 20 \
  --save-interval 500 \
  --checkpoint-dir checkpoints \
  --generation-interval 500 \
  --seed 1337
```

The default prompt is `Once upon a time`. At each checkpoint the demo prints
three decoders using the same model weights: greedy decoding, temperature
sampling with temperature `0.8`, and top-k sampling with `k=40`. Sampling uses
fixed per-checkpoint seeds for reproducibility. Generation is limited to 40 new
tokens. The default run uses 1000 training steps, which produces comparisons at
steps 0, 500, and 1000.

The same command saves `checkpoints/step_000500.pt`, numbered checkpoints at
later save steps, and `checkpoints/latest.pt`. Resume to a final global step
with:

```bash
python scripts/train_generate.py \
  --steps 3000 \
  --resume checkpoints/latest.pt \
  --eval-interval 500 \
  --eval-batches 20 \
  --save-interval 500 \
  --checkpoint-dir checkpoints \
  --generation-interval 500 \
  --seed 1337
```

Validation runs under `torch.no_grad()`, uses a fixed random sample from the
full validation stream, and restores the model's training mode afterwards.
Checkpoints contain model and optimizer state, global step, model/data/training
configuration, and the random states needed to continue sampling batches.
