from baseline_sampling import generate_text
from data_utils import load_prompts_and_references
from reward_functions import (
    compute_fluency_score,
    compute_coherence_score,
    compute_diversity_score,
)


def evaluate_sampling(prompts, references):
    """Evaluate model sampling based on fluency, coherence, and diversity."""
    generated_texts = []

    for prompt in prompts:
        generated = generate_text(prompt)
        generated_texts.append(generated)

    # Metrics
    fluency_scores = [compute_fluency_score(g) for g in generated_texts]
    coherence_scores = [
        compute_coherence_score(r, g) for r, g in zip(references, generated_texts)
    ]
    diversity_score = compute_diversity_score(generated_texts)

    print("\n=== Evaluation Results ===")
    print(f"Avg Fluency Score: {sum(fluency_scores)/len(fluency_scores):.4f}")
    print(
        f"Avg Coherence Score (BLEU): {sum(coherence_scores)/len(coherence_scores):.4f}"
    )
    print(f"Diversity Score (1-SelfBLEU): {diversity_score:.4f}")

    # Print some generated examples
    for i in range(5):
        print(f"\nPrompt: {prompts[i]}")
        print(f"Generated: {generated_texts[i]}")
        print(f"Reference: {references[i]}")


if __name__ == "__main__":
    _, (prompts, references) = load_prompts_and_references(test_size=1000)
    evaluate_sampling(prompts, references)
