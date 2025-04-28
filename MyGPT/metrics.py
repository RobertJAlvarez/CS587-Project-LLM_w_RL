import nltk  # type: ignore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
from rouge_score import rouge_scorer  # type: ignore

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
    return score["rougeL"].fmeasure  # Return F1 measure


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
