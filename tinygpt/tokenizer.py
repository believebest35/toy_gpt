"""BPE tokenizer training and text conversion utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
SPECIAL_TOKENS = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]
DEFAULT_VOCAB_SIZE = 8192


def train_tokenizer(
    text_iterator: Iterable[str], vocab_size: int = DEFAULT_VOCAB_SIZE
) -> Tokenizer:
    """Train a ByteLevel BPE tokenizer from an iterable of text examples."""

    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise ValueError("vocab_size must be an integer")
    if vocab_size < len(SPECIAL_TOKENS):
        raise ValueError(
            f"vocab_size must be at least {len(SPECIAL_TOKENS)} for the special tokens"
        )

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),
    )
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: str | Path) -> Path:
    """Save a tokenizer using the library's native JSON format."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return output_path


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a tokenizer from a native ``tokenizer.json`` file."""

    return Tokenizer.from_file(str(Path(path)))


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode text without implicitly adding BOS or EOS tokens."""

    encoding = tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in encoding.ids]


def decode(tokenizer: Tokenizer, ids: Iterable[int]) -> str:
    """Decode token IDs into text without dropping special tokens implicitly."""

    return tokenizer.decode(
        [int(token_id) for token_id in ids],
        skip_special_tokens=False,
    )


__all__ = [
    "BOS_TOKEN",
    "DEFAULT_VOCAB_SIZE",
    "EOS_TOKEN",
    "PAD_TOKEN",
    "SPECIAL_TOKENS",
    "UNK_TOKEN",
    "decode",
    "encode",
    "load_tokenizer",
    "save_tokenizer",
    "train_tokenizer",
]
