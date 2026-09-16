# TinyGPT

TinyGPT is an educational implementation of a GPT-style decoder-only language
model. The project is intended to make the mechanics of language-model
pretraining explicit and easy to inspect.

The long-term project will cover tokenizer training, language-model dataset
preparation, causal attention, a Transformer implementation, next-token
pretraining, text generation, and evaluation. The model will be built from
basic PyTorch primitives rather than a pretrained GPT implementation.

The repository currently contains Phase 1 infrastructure and the Phase 2
text-to-token pipeline. Phase 2 provides TinyStories access, custom BPE
tokenizer training, tokenizer serialization, and encoding/decoding utilities.
Model training and tokenized dataset preprocessing have not been implemented.

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
`artifacts/tokenizer/tokenizer.json`. Tokenized dataset preparation and model
training belong to later phases.
