import nltk
from nltk.translate.bleu_score import sentence_bleu

nltk.download("punkt")


def compute_fluency_score(text):
    """Simple length-based fluency (better: use a small language model later)."""
    words = nltk.word_tokenize(text)
    if len(words) < 5:
        return -1.0  # Penalize very short text.
    return 1.0


def compute_coherence_score(reference, generated):
    """Use BLEU score for rough coherence."""
    reference_tokens = nltk.word_tokenize(reference)
    generated_tokens = nltk.word_tokenize(generated)
    return sentence_bleu([reference_tokens], generated_tokens)


def compute_diversity_score(samples) -> float:
    """Self-BLEU — lower means more diverse."""
    scores = []
    for i, sample in enumerate(samples):
        others = samples[:i] + samples[i + 1 :]
        scores.extend(
            [
                sentence_bleu([nltk.word_tokenize(other)], nltk.word_tokenize(sample))
                for other in others
            ]
        )
    return 1.0 - sum(scores) / len(scores)
