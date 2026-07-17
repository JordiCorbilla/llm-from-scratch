import pytest

from frankengpt.tokenizer import CharTokenizer, WordTokenizer


def test_tokenizer_round_trip_and_persistence(tmp_path):
    tokenizer = CharTokenizer.from_text("hello world!")
    assert tokenizer.decode(tokenizer.encode("hello!")) == "hello!"
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    assert CharTokenizer.load(path).tokens == tokenizer.tokens


def test_tokenizer_rejects_unknown_characters():
    with pytest.raises(ValueError, match="not in tokenizer"):
        CharTokenizer.from_text("abc").encode("d")


def test_word_tokenizer_emits_word_ids_and_readable_text():
    tokenizer = WordTokenizer.from_text("Hello, world! Hello again.", max_vocab=8)
    assert tokenizer.decode(tokenizer.encode("Hello, world!")) == "Hello, world!"
    assert tokenizer.vocab_size <= 8
