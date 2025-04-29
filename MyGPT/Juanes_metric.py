import torch  # type: ignore
import math
import tiktoken  # type: ignore

from my_gpt import GPT, Config


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    block_size: int = 128,
    stride: int = 64,
    device: torch.device = None,
) -> float:
    """
    Compute PPL on `text` by scoring at most `block_size` tokens at a time,
    advancing by `stride` tokens per window.
    """
    device = device or next(model.parameters()).device
    model.eval()

    # Full token IDs.
    enc = tokenizer.encode(text)
    n = len(enc)
    if n < 2:
        return float("inf")

    total_logp = 0.0
    total_count = 0

    for start in range(0, n - 1, stride):
        end = min(start + block_size, n)
        input_ids = torch.tensor(enc[start:end], device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids)[0]  # (1, L, V)
            log_probs = torch.log_softmax(logits, dim=-1)

        # We score tokens from start+1 to end.
        for i in range(start + 1, end):
            token_id = enc[i]
            pos = i - start  # Position in this window.
            total_logp += log_probs[0, pos - 1, token_id].item()
            total_count += 1

        if end == n:
            break

    # average and exponentiate
    avg_neg_logp = -total_logp / total_count
    return math.exp(avg_neg_logp)


def distinct_n(tokens: list[str], n: int) -> float:
    """
    Compute Distinct-n = (# unique n-grams) / (# total n-grams).
    """
    if len(tokens) < n:
        return 0.0
    ngrams = zip(*(tokens[i:] for i in range(n)))
    ngrams_list = list(ngrams)
    if not ngrams_list:
        return 0.0
    unique = len(set(ngrams_list))
    return unique / len(ngrams_list)


def repetition_rate(tokens: list[str]) -> float:
    """
    Fraction of tokens that are repeats:
      (total tokens - unique tokens) / total tokens.
    """
    total = len(tokens)
    if total == 0:
        return 0.0
    unique = len(set(tokens))
    return (total - unique) / total


def evaluate_text(model: torch.nn.Module, tokenizer, text: str) -> dict[str, float]:
    """
    Compute all metrics on a single generated string `text`.
    Returns a dict with keys:
      - perplexity
      - distinct1
      - distinct2
      - repetition_rate
    """
    # 1) Perplexity
    ppl = compute_perplexity(model, tokenizer, text)

    # 2) Tokenize on whitespace for n-grams & repetition
    words = text.split()

    # 3) Diversity metrics
    d1 = distinct_n(words, 1)
    d2 = distinct_n(words, 2)

    # 4) Repetition
    rep = repetition_rate(words)

    return {
        "perplexity": ppl,
        "distinct1": d1,
        "distinct2": d2,
        "repetition_rate": rep,
    }


if __name__ == "__main__":
    # quick smoke test
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model configuration (must match the trained model's configuration)
    config = Config(
        vocab_size=tokenizer.n_vocab, n_embd=256, n_head=8, n_layer=4, block_size=128
    )

    # Initialize the model
    model = GPT(config)
    sample = "In a hole in the ground there lived a hobbit."

    metrics = evaluate_text(model, tokenizer, sample)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
