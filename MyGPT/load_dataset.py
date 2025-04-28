from datasets import load_dataset  # type: ignore
import random


def load_prompts_and_references(train_pct: int = 10, test_pct: str = 2):
    """
    Load prompts and references from OpenWebText.
    Split into training and test sets.
    """
    tot_pct = train_pct + test_pct
    assert tot_pct <= 100, "Train + test percent > 100"

    print("Loading datasets...")
    openwebtext = load_dataset("stas/openwebtext-10k")

    texts = [sample["text"] for sample in openwebtext["train"] if "text" in sample]

    print(f"Total samples: {len(texts)}")
    random.shuffle(texts)

    prompts = []
    references = []
    for text in texts:
        sentences = text.split(".")
        if len(sentences) >= 2:
            prompts.append(sentences[0].strip() + ".")
            references.append(". ".join(sentences[:2]).strip() + ".")

    # Split.
    train_size = int((train_pct / tot_pct) * len(texts))

    train_prompts = prompts[:train_size]
    train_references = references[:train_size]
    test_prompts = prompts[train_size:]
    test_references = references[train_size:]

    print(
        f"Loaded {len(train_prompts)} training samples and {len(test_prompts)} test samples."
    )
    return (train_prompts, train_references), (test_prompts, test_references)
