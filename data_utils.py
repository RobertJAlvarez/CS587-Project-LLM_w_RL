from datasets import load_dataset  # type: ignore
import random


def load_prompts_and_references(train_size=1000, test_size=100):
    """
    Load prompts and references from OpenWebText and The Pile.
    Split into training and test sets.
    """
    print("Loading datasets...")
    openwebtext = load_dataset("openwebtext", split="train[:1%]")
    pile = load_dataset("EleutherAI/pile", split="train[:1%]")

    combined_texts = []
    for sample in openwebtext:
        if "text" in sample:
            combined_texts.append(sample["text"])
    for sample in pile:
        if "text" in sample:
            combined_texts.append(sample["text"])

    print(f"Total combined samples: {len(combined_texts)}")
    random.shuffle(combined_texts)

    prompts = []
    references = []
    for text in combined_texts:
        sentences = text.split(".")
        if len(sentences) >= 2:
            prompt = sentences[0].strip()
            reference = ". ".join(sentences[:2]).strip() + "."
            prompts.append(prompt)
            references.append(reference)
        if len(prompts) >= (train_size + test_size):
            break

    # Split
    train_prompts = prompts[:train_size]
    train_references = references[:train_size]
    test_prompts = prompts[train_size : train_size + test_size]
    test_references = references[train_size : train_size + test_size]

    print(
        f"Loaded {len(train_prompts)} training samples and {len(test_prompts)} test samples."
    )
    return (train_prompts, train_references), (test_prompts, test_references)
