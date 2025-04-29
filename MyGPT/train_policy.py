import argparse
import tiktoken  # type: ignore
import time
import torch  # type: ignore
import torch.optim as optim  # type: ignore

from generate_w_policy import generate_w_policy
from load_dataset import sample_paragraph_splits, load_prompts_and_references
from metrics import compute_fluency_score, compute_coherence_score
from my_gpt import GPT, Config
from sampling_policy import SamplingPolicy
from test_policy import test_policy
from utils import normalize, ceildiv


def compute_rewards(sample, reference) -> float:
    fluency = compute_fluency_score(sample)
    coherence = compute_coherence_score(reference, sample)
    return 0.6 * fluency + 0.4 * coherence


def compute_pseudo_log_prob(
    temperature: torch.Tensor, top_p: torch.Tensor, alpha=1.0, beta=1.0
) -> torch.Tensor:
    return -alpha * temperature + beta * (1.0 - top_p)


def compute_entropy_regularizer(
    temperature: torch.Tensor, top_p: torch.Tensor, temp_weight=1.0, top_p_weight=1.0
) -> torch.Tensor:
    """
    Penalize overly confident (low temperature) or overly greedy (low top-p) behavior.
    """
    # Encourage higher temp.
    temp_entropy = -(temperature * torch.log(temperature + 1e-8))
    # Encourage higher top_p.
    top_p_entropy = -(top_p * torch.log(top_p + 1e-8))
    return temp_weight * temp_entropy + top_p_weight * top_p_entropy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with a trained RoPE GPT model"
    )
    parser.add_argument(
        "--in_distribution",
        type=bool,
        default=True,
        help="Load data from module training text OR use a different dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of epochs to train for"
    )
    args = parser.parse_args()

    # Hyperparameters.
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    # Load the tokenizer (GPT-4 tokenizer).
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Set device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model configuration (must match the trained model's configuration).
    config = Config(
        vocab_size=tokenizer.n_vocab, n_embd=256, n_head=8, n_layer=4, block_size=128
    )

    # Initialize the model.
    model = GPT(config)

    # Load the trained weights.
    try:
        model_path = "models/best_gpt_model.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Move model to the appropriate device.
    model.to(device)
    model.eval()  # Set the model to evaluation mode.

    # Generate text using specified setting.
    print("Training policy using 'KV cache'")

    # Initialize policy.
    policy = SamplingPolicy().to(device)

    # Load training and testing prompts.
    prompts = []
    references = []
    ks = [5, 10, 15, 20]
    n_samples = ceildiv(num_epochs * batch_size, len(ks))

    # Select function to get data In-Data-Distribution or Out-of-Data-Distribution.
    gen_dataset = (
        sample_paragraph_splits if args.in_distribution else load_prompts_and_references
    )

    for k in ks:
        ps, rs = gen_dataset(num_samples=n_samples, k=k)
        prompts.extend(ps)
        references.extend(rs)

    print(f"Loaded {len(prompts)} prompt-reference pairs.")
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)  # AdamW w/ lr=5e-4?

    # Early stopping variables.
    best_avg_reward = -float("inf")
    best_policy_path = "models/best_policy.pt"

    start_time = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        rewards = []
        log_probs = []
        entropies = []
        st = epoch * batch_size
        ed = (epoch + 1) * batch_size
        for prompt, reference in zip(prompts[st:ed], references[st:ed]):
            # Generate text from prompt.
            sample, temperature, top_p = generate_w_policy(
                model, prompt, tokenizer, policy
            )

            rewards.append(compute_rewards(sample, reference))
            log_probs.append(compute_pseudo_log_prob(temperature, top_p))
            entropies.append(compute_entropy_regularizer(temperature, top_p))

        rewards = torch.tensor(rewards, device=device)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        # Policy gradient loss: maximize expected reward + mean(entropy).
        loss = (normalize(rewards) * log_probs).mean() - 0.01 * entropies.mean()

        loss.backward()
        optimizer.step()

        avg_reward = rewards.mean().item()

        print(f"Epoch {epoch+1}: Loss {loss.item():.4f} Reward {avg_reward:.4f}")

        # Save best policy.
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(policy.state_dict(), best_policy_path)

    print(f"Total training time = {(time.time() - start_time):.4f}s")

    test_policy()
