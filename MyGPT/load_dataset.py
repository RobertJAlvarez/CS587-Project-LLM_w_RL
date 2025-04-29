from datasets import load_dataset  # type: ignore
import random
import re


def tokenize(selected: list[str], k: int) -> tuple[list[str], list[str]]:
    selected_tokens = [text.split()[:k] for text in selected]
    prompts = [" ".join(tokens) for tokens in selected_tokens]
    return prompts, selected


def load_prompts_and_references(num_samples: int = 10, k: int = 5) -> tuple[list, list]:
    """
    Load prompts and references from OpenWebText.
    Split into training and test sets.
    """
    print("Loading datasets...")
    openwebtext = load_dataset("stas/openwebtext-10k")

    # Extract each prompt.
    org_text = [sample["text"] for sample in openwebtext["train"] if "text" in sample]

    # prompts = []
    # references = []
    # for prompt in org_text:
    #     sentences = prompt.split(".")

    #     # Don't use single sentece prompts.
    #     if len(sentences) <= 1:
    #         continue

    #     # Use from 1-10 sentences as prompt.
    #     n_sentences = len(sentences)
    #     idx = n_sentences - 1 if n_sentences <= 7 else 7
    #     next_sentence_words = sentences[idx].lstrip().split(" ")
    #     if len(next_sentence_words) <= 1:
    #         continue
    #     prompts.append(". ".join(sentences[:idx]) + ". " + next_sentence_words[0] + " ")
    #     references.append(prompt)

    return tokenize(org_text[:num_samples], k)


def sample_paragraph_splits(
    file_path: str = "input.txt", num_samples: int = 10, k: int = 5
) -> tuple[list[str], list[str]]:
    """
    Read the text from `file_path`, split into paragraphs, randomly select
    `num_samples` of them, and for each:
      - Append the EOT marker to the paragraph text
      - Split on whitespace into tokens
      - Take the first k tokens into X
      - Take all remaining tokens (from k up to and including EOT) into Y

    Returns:
      X: list of length num_samples; each is a list of k tokens (strings)
      Y: list of length num_samples; each is a list of the remaining tokens
    """
    # 1) Load and split into paragraphs (blocks separated by blank lines)
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]

    # 2) Sample paragraphs (without replacement if possible)
    if len(paragraphs) >= num_samples:
        selected = random.sample(paragraphs, num_samples)
    else:
        selected = [random.choice(paragraphs) for _ in range(num_samples)]

    return tokenize(selected, k)
