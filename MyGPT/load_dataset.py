from datasets import load_dataset  # type: ignore
import random
import re
from datetime import datetime


def tokenize(selected: list[str], k: int) -> tuple[list[str], list[str]]:
    selected_tokens = [text.split()[:k] for text in selected]
    prompts = [" ".join(tokens) for tokens in selected_tokens]
    return prompts, selected


def sample_prompts(prompts: list[str], num_samples: int) -> list[str]:
    # Sample paragraphs (without replacement if possible).
    random.seed(datetime.now().timestamp())
    if len(prompts) >= num_samples:
        selected = random.sample(prompts, num_samples)
    else:
        selected = [random.choice(prompts) for _ in range(num_samples)]
    return selected


def load_prompts_and_references(num_samples: int = 10, k: int = 5) -> tuple[list, list]:
    """
    Load prompts and references from OpenWebText.
    Split into training and test sets.
    """
    print("Loading datasets...")
    openwebtext = load_dataset("stas/openwebtext-10k")

    # Extract each text.
    org_text = [sample["text"] for sample in openwebtext["train"] if "text" in sample]

    # Sample texts.
    sample_text = sample_prompts(org_text, num_samples)

    # Devide into prompt and ground truth.
    return tokenize(sample_text, k)


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

    # Sample paragraphs.
    selected = sample_prompts(paragraphs, num_samples)

    return tokenize(selected, k)
