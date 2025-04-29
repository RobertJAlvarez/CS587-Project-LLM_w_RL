import torch  # type: ignore


def ceildiv(a, b):
    return -(a // -b)


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def top_p_filtering(logits, top_p: float = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[0, indices_to_remove] = -float("Inf")
    return logits


def split_dataset(
    X: list[str], Y: list[str], train_pct: int = 80
) -> tuple[tuple[str, str], tuple[str, str]]:
    # Slip into training and testing.
    train_size = int((train_pct / 100) * len(X))

    train_prompts = X[:train_size]
    train_references = Y[:train_size]
    test_prompts = X[train_size:]
    test_references = Y[train_size:]

    return (train_prompts, train_references), (test_prompts, test_references)
