import argparse
import tiktoken  # type: ignore
import time
import torch  # type: ignore
import torch.optim as optim  # type: ignore

from my_gpt import GPT, Config
from reward_logger import RewardLogger
from utils import normalize
from metrics import print_metrics, compute_fluency_score, compute_coherence_score
from load_dataset import sample_paragraph_splits
from sampling_policy import SamplingPolicy
from generate_w_policy import generate_w_policy


def split_dataset(X, Y, train_pct: int = 80):
    # Slip into training and testing.
    train_size = int((train_pct / 100) * len(X))

    train_prompts = X[:train_size]
    train_references = Y[:train_size]
    test_prompts = X[train_size:]
    test_references = Y[train_size:]

    return (train_prompts, train_references), (test_prompts, test_references)


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
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=12, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--k", type=int, default=5, help="First k tokens from paragraph"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of paragraph samples"
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
    (train_prompts, train_references), (test_prompts, test_references) = split_dataset(
        *sample_paragraph_splits(k=args.k, num_samples=args.num_samples), train_pct=80
    )

    print(f"Loaded {len(train_prompts)} prompt-reference pairs.")
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)  # AdamW w/ lr=5e-4?

    logger = RewardLogger()

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
        for prompt, reference in zip(train_prompts[st:ed], train_references[st:ed]):
            # Generate text from prompt.
            sample, temperature, top_p = generate_w_policy(
                model, tokenizer, prompt, policy
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
        logger.log(avg_reward)

        print(f"Epoch {epoch+1}: Loss {loss.item():.4f} Reward {avg_reward:.4f}")

        # Save best policy.
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(policy.state_dict(), best_policy_path)

    logger.save()
    print(f"Total training time = {(time.time() - start_time):.4f}s")

    # Evaluate model sampling based on fluency, coherence, and diversity.

    # Generate text w/o policy + metrics.
    start_time = time.time()
    gen_text_wo_policy = []
    with torch.no_grad():
        for prompt in test_prompts:
            sample, *_ = generate_w_policy(model, tokenizer, prompt)
            gen_text_wo_policy.append(sample)
    print_metrics(gen_text_wo_policy, test_references, w_policy=False)
    print(f"Total gen w/o policy time = {(time.time() - start_time):.4f}s")

    # Load the best policy for evaluation.
    policy.load_state_dict(torch.load(best_policy_path))

    # Generate text w/ policy + metrics.
    gen_text_w_policy = []
    start_time = time.time()
    with torch.no_grad():
        for prompt in test_prompts:
            sample, *_ = generate_w_policy(model, tokenizer, prompt, policy)
            gen_text_w_policy.append(sample)
    print_metrics(gen_text_w_policy, test_references, w_policy=True)
    print(f"Total gen w/ policy time = {(time.time() - start_time):.4f}s")

    # Print some generated examples.
    for i in range(3):
        print("======================")
        print(f"** Prompt **\n{test_prompts[i]}")
        print(f"** Gen w/ policy **\n{gen_text_w_policy[i]}")
        print(f"** Gen w/o policy **\n{gen_text_wo_policy[i]}")
        print(f"** Reference **\n{test_references[i]}")
        print("======================")
