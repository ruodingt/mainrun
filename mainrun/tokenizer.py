
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers


def apply_cap_tags(text: str) -> str:
    """Preprocess text for cap-tag tokenization.

    Inserts <cap> before each word that starts with an uppercase letter, then
    lowercases everything. The tokenizer must have <cap> registered as a special
    token so BPE never splits it.

    Example: "Show HN: My Tool" -> "<cap> show <cap> hn: my <cap> tool"
    """
    parts = []
    for word in text.split(' '):
        if word and word[0].isupper():
            parts.append('<cap>')
        parts.append(word.lower())
    return ' '.join(parts)


def train_tokenizer(
    titles: list[str],
    vocab_size: int,
    unk_token: str = "<unk>",
    pad_token: str = "<pad>",
    eos_token: str = "<eos>",
    lowercase: bool = False,
    cap_tags: bool = False,
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    # Lowercase normalizer is injected before pre-tokenization so the vocabulary
    # is built entirely on lowercased text. At encode time the same normalizer
    # fires automatically, so no caller-side lowercasing is needed.
    if lowercase:
        tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    cap_token = "<cap>"
    special_tokens = [pad_token, eos_token, unk_token]
    if cap_tags:
        special_tokens.append(cap_token)
        titles = [apply_cap_tags(t) for t in titles]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self): return self.tk.get_vocab_size()


def load_or_train_tokenizer(
    titles: list[str],
    vocab_size: int,
    dataset: str,
    seed: int,
    num_titles: int,
    val_frac: float,
    cache_dir: str = "./data",
    **kwargs,
) -> "BPETokenizer":
    """Load a cached tokenizer or train and cache a new one.

    Cache key covers every parameter that determines the tokenizer's vocabulary:
    dataset identity, data split config, vocab size, and tokenizer options.
    """
    key = hashlib.md5(
        json.dumps(
            {"dataset": dataset, "vocab_size": vocab_size, "seed": seed,
             "num_titles": num_titles, "val_frac": val_frac, **kwargs},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    cache_path = Path(cache_dir) / f"tok_{key}.json"
    if cache_path.exists():
        print(f"tokenizer: loading from cache {cache_path}")
        return BPETokenizer(Tokenizer.from_file(str(cache_path)))
    print("tokenizer: training (no cache hit)...")
    tok_raw = train_tokenizer(titles, vocab_size, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tok_raw.save(str(cache_path))
    print(f"tokenizer: saved to {cache_path}")
    return BPETokenizer(tok_raw)
