from __future__ import annotations

from pathlib import Path

from tinygpt.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    decode,
    encode,
    load_tokenizer,
    save_tokenizer,
    train_tokenizer,
)


CORPUS = [
    "Once upon a time there was a little girl.",
    "The little dog ran home.",
    "Tom and Alice played in the garden.",
    "The dog found a red ball and wagged its tail.",
    "Alice read a book before going to sleep.",
]


def make_tokenizer():
    return train_tokenizer((text for text in CORPUS), vocab_size=512)


def test_train_tokenizer_from_in_memory_iterator() -> None:
    tokenizer = make_tokenizer()

    assert tokenizer.get_vocab_size() > 0


def test_encode_returns_integer_ids() -> None:
    tokenizer = make_tokenizer()

    ids = encode(tokenizer, "Once upon a time.")

    assert isinstance(ids, list)
    assert all(isinstance(token_id, int) for token_id in ids)


def test_decode_produces_readable_text() -> None:
    tokenizer = make_tokenizer()
    text = "The little dog ran home."

    decoded = decode(tokenizer, encode(tokenizer, text))

    assert "little dog" in decoded
    assert decoded.strip().endswith("home.")


def test_all_special_tokens_exist() -> None:
    tokenizer = make_tokenizer()

    for token in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN):
        assert tokenizer.token_to_id(token) is not None


def test_save_and_load_preserve_encoding(tmp_path: Path) -> None:
    tokenizer = make_tokenizer()
    text = "Once upon a time there was a little girl."
    path = tmp_path / "tokenizer.json"

    save_tokenizer(tokenizer, path)
    reloaded = load_tokenizer(path)

    assert path.is_file()
    assert encode(reloaded, text) == encode(tokenizer, text)


def test_save_and_load_preserve_special_token_ids(tmp_path: Path) -> None:
    tokenizer = make_tokenizer()
    path = tmp_path / "tokenizer.json"

    save_tokenizer(tokenizer, path)
    reloaded = load_tokenizer(path)

    for token in SPECIAL_TOKENS:
        assert reloaded.token_to_id(token) == tokenizer.token_to_id(token)


def test_special_token_list_is_canonical() -> None:
    assert SPECIAL_TOKENS == [
        "<|bos|>",
        "<|eos|>",
        "<|pad|>",
        "<|unk|>",
    ]


def test_encode_does_not_prepend_bos() -> None:
    tokenizer = make_tokenizer()
    ids = encode(tokenizer, "Once upon a time.")
    bos_id = tokenizer.token_to_id(BOS_TOKEN)

    assert ids
    assert ids[0] != bos_id


def test_encode_does_not_append_eos() -> None:
    tokenizer = make_tokenizer()
    ids = encode(tokenizer, "Once upon a time.")
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    assert ids
    assert ids[-1] != eos_id
