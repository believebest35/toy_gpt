# TinyGPT

TinyGPT is an educational implementation of a GPT-style decoder-only language
model. The project is intended to make the mechanics of language-model
pretraining explicit and easy to inspect.

The long-term project will cover tokenizer training, language-model dataset
preparation, causal attention, a Transformer implementation, next-token
pretraining, text generation, and evaluation. The model will be built from
basic PyTorch primitives rather than a pretrained GPT implementation.

The repository currently contains only the Phase 1 infrastructure: project
setup, typed configuration loading, configuration validation, and tests.

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
