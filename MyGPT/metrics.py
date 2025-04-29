import math
import nltk  # type: ignore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
from rouge_score import rouge_scorer  # type: ignore
import torch  # type: ignore


# Download NLTK resources if not already present
nltk.download("punkt")


def compute_fluency_score(text) -> float:
    """
    A simple proxy for fluency: longer sentences, proper punctuation.
    (Optional improvement: use perplexity from a small language model.)
    """
    sentences = nltk.tokenize.sent_tokenize(text)
    if not sentences:
        return 0.0

    avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(
        sentences
    )

    # Normalize to a range, e.g., divide by 20 (typical sentence length).
    return min(avg_sentence_length / 20.0, 1.0)


def compute_coherence_score(reference, generated):
    """
    Use BLEU score between reference and generated text.
    """
    reference_tokens = nltk.tokenize.word_tokenize(reference.lower())
    generated_tokens = nltk.tokenize.word_tokenize(generated.lower())
    return sentence_bleu(
        [reference_tokens],
        generated_tokens,
        smoothing_function=SmoothingFunction().method4,
    )


def compute_diversity_score(samples) -> float:
    """
    Compute Self-BLEU: lower is better (more diversity).
    """
    n = len(samples)

    # Perfect diversity if only one sample
    if n <= 1:
        return 1.0

    tokenized_samples = [
        nltk.tokenize.word_tokenize(sample.lower()) for sample in samples
    ]
    total_self_bleu = sum(
        [
            sentence_bleu(
                tokenized_samples[:i] + tokenized_samples[i + 1 :],
                tokenized_samples[i],
                smoothing_function=SmoothingFunction().method4,
            )
            for i in range(n)
        ]
    )

    return 1.0 - (total_self_bleu / n)


def compute_rouge_l(reference: str, generated: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, generated)
    return score["rougeL"].fmeasure  # Return F1 measure.


def print_metrics(
    gen_prompts: list[str], test_references: list[str], w_policy: bool
) -> None:
    fluency_scores = [compute_fluency_score(g) for g in gen_prompts]
    coherence_scores = [
        compute_coherence_score(r, g) for r, g in zip(test_references, gen_prompts)
    ]
    diversity_score = compute_diversity_score(gen_prompts)
    rouge_l_scores = [
        compute_rouge_l(r, g) for r, g in zip(test_references, gen_prompts)
    ]

    w_wo_str = "" if w_policy else "out"
    print(f"\n=== Evaluation Results With{w_wo_str} Policy ===")
    print(f"Avg Fluency Score: {sum(fluency_scores)/len(fluency_scores):.4f}")
    print(
        f"Avg Coherence Score (BLEU): {sum(coherence_scores)/len(coherence_scores):.4f} (higher is better)"
    )
    print(f"Avg ROUGE-L Score: {sum(rouge_l_scores)/len(rouge_l_scores):.4f}")
    print(f"Diversity Score (1-SelfBLEU): {diversity_score:.4f} (lower is better)")


## Juanes metrics.


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


def print_text_evaluation(model: torch.nn.Module, tokenizer, text: str) -> None:
    """
    Compute all metrics on a single generated string `text`.
    Print:
      - perplexity
      - distinct1
      - distinct2
      - repetition_rate
    """
    ppl = compute_perplexity(model, tokenizer, text)

    # Tokenize on whitespace for n-grams & repetition
    words = text.split()
    d1 = distinct_n(words, 1)
    d2 = distinct_n(words, 2)
    rep = repetition_rate(words)

    titles = ["perplexity", "distinct1", "distinct2", "repetition_rate"]
    scores = [ppl, d1, d2, rep]
    for k, v in zip(titles, scores):
        print(f"{k}: {v:.4f}")
