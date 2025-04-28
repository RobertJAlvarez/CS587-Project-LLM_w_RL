import argparse
import tiktoken  # type: ignore
import time
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore

from my_gpt import GPT, Config
from reward_logger import RewardLogger
from utils import top_p_filtering, normalize
from metrics import print_metrics, compute_fluency_score, compute_coherence_score
from load_dataset import load_prompts_and_references


def generate(
    model,
    tokenizer,
    prompt,
    other_eots: list = [],
    policy=None,
    max_new_tokens=1000,
    default_top_p=0.9,
) -> str:
    """
    Generate text from a prompt using the trained GPT model with KV caching support.

    Args:
        model: The trained GPT model.
        tokenizer: The tokenizer used to encode/decode text.
        prompt: The text prompt to start generation.
        policy: Policy to generate the temperature to be use for the next token.
        max_new_tokens: Maximum number of tokens to generate.
        default_top_p: Consider all token with high probability where its comulative is at least default_top_p.

    Returns:
        The generated text including the prompt and generation time.
    """
    # Encode the prompt.
    encoded_prompt = tokenizer.encode(prompt)
    tokens = (
        torch.tensor(encoded_prompt, dtype=torch.long)
        .unsqueeze(0)
        .to(model.lm_head.weight.device)
    )

    if policy:
        temperature, top_p = policy(model.transformer.wte(tokens))
    else:
        temperature = 0.5
        top_p = default_top_p

    # Initialize the past key values to None (no caching yet).
    past_key_values = None
    max_past_key_values_len = model.config.block_size - 1

    # Create set of valid end of token.
    eots = set(other_eots + [tokenizer.eot_token])

    # Generate tokens one at a time.
    for _ in range(max_new_tokens):
        # For KV cache: after first iteration, only process the last token.
        if past_key_values is None:
            # Get only the last block_size tokens if input is too long.
            context = tokens[:, -model.config.block_size :]
        else:
            context = tokens[:, -1:]  # With KV cache, we only need the last token.
            # Get only the last block_size - 1 KV cache if the total input (KV cache + context) is too long.
            if past_key_values[0][0].size(2) > max_past_key_values_len:
                past_key_values = list(
                    tuple(t[:, :, -max_past_key_values_len:] for t in layer_past)
                    for layer_past in past_key_values
                )

        # Forward pass to get logits.
        with torch.no_grad():
            logits, new_past_key_values = model(
                context, past_key_values=past_key_values
            )

            # Update KV cache for next iteration if using cache.
            past_key_values = new_past_key_values

        # Focus on the last token's predictions.
        logits = logits[:, -1, :] / temperature

        # Filter tokens to higher prop with compulative prob of top_p.
        logits = top_p_filtering(logits, top_p=top_p)

        # Apply softmax to get probabilities.
        probs = torch.softmax(logits, dim=-1)

        # Sample from the distribution.
        next_token = torch.multinomial(probs, num_samples=1)

        # If we reach the end of text token, stop.
        if next_token.item() in eots:
            break
        else:
            # Append the token to our sequence.
            tokens = torch.cat((tokens, next_token), dim=1)

        # Update temperature if policy was given.
        if policy:
            with torch.no_grad():
                temperature, top_p = policy(model.transformer.wte(tokens))

    # Decode the tokens.
    generated_text = tokenizer.decode(tokens[0].tolist())

    # Return both the generated text and timing information.
    return generated_text, temperature, top_p


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, 1)

    def forward(self, token_embeddings):
        # token_embeddings: (batch_size, seq_len, embed_dim)
        attn_weights = torch.softmax(self.query(token_embeddings), dim=1)
        return (attn_weights * token_embeddings).sum(dim=1)  # (batch_size, embed_dim)


class SamplingPolicy(nn.Module):
    """Learn temperature and top-p adjustments based on prompt embeddings."""

    def __init__(self, embed_dim=256) -> None:
        super().__init__()
        self.attention_pooling = AttentionPooling(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Now output 2 values: [temperature, top_p]
        )

    def forward(self, token_embeddings) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_emb = self.attention_pooling(token_embeddings)  # (batch_size, embed_dim)
        outputs = self.mlp(pooled_emb)
        temperature = torch.sigmoid(outputs[:, 0]) * 2.0  # (0, 2)
        top_p = 0.8 + torch.sigmoid(outputs[:, 1]) * 0.2  # (0.8, 1.0)
        return temperature, top_p


def compute_rewards(sample, reference) -> float:
    fluency = compute_fluency_score(sample)
    coherence = compute_coherence_score(reference, sample)
    return 0.6 * fluency + 0.4 * coherence


def compute_pseudo_log_prob(
    temperature: torch.Tensor, top_p: torch.Tensor, alpha=1.0, beta=1.0
) -> torch.Tensor:
    """
    Smarter pseudo-log-prob using both temperature and top-p.
    """
    return -alpha * temperature + beta * (1.0 - top_p)


def compute_entropy_regularizer(
    temperature: torch.Tensor, top_p: torch.Tensor, temp_weight=1.0, top_p_weight=1.0
) -> torch.Tensor:
    """
    Penalize overly confident (low temperature) or overly greedy (low top-p) behavior.
    """
    temp_entropy = -(
        temperature * torch.log(temperature + 1e-8)
    )  # encourage higher temp
    top_p_entropy = -(top_p * torch.log(top_p + 1e-8))  # encourage higher top_p
    return temp_weight * temp_entropy + top_p_weight * top_p_entropy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with a trained RoPE GPT model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_gpt_model.pt",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
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
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")
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
    train, test = load_prompts_and_references()
    train_prompts, train_references = train

    print(f"Loaded {len(train_prompts)} prompt-reference pairs.")
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)  # AdamW w/ lr=5e-4?

    logger = RewardLogger()

    # Testing is done between the real second sentence and the generated one.
    #   So, a period is a valid end of token.
    other_eots = tokenizer.encode(".")

    # Early stopping variables.
    best_avg_reward = -float("inf")
    patience = 5
    no_improve_epochs = 0

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
            sample, temperature, top_p = generate(
                model, tokenizer, prompt, other_eots, policy
            )

            rewards.append(compute_rewards(sample, reference))
            log_probs.append(compute_pseudo_log_prob(temperature, top_p))
            entropies.append(compute_entropy_regularizer(temperature, top_p))

        rewards = torch.tensor(rewards, device=device)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        # Normalize rewards: (r - mean) / (std + 1e-6) for stability
        rewards = normalize(rewards)

        # Policy gradient loss: maximize expected reward + mean(entropy).
        loss = (rewards * log_probs).mean() - 0.01 * entropies.mean()

        loss.backward()
        optimizer.step()

        avg_reward = rewards.mean().item()
        logger.log(avg_reward)

        print(f"Epoch {epoch+1}: Loss {loss.item():.4f} Reward {avg_reward:.4f}")

        # Save best policy.
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            no_improve_epochs = 0
            torch.save(policy.state_dict(), "models/best_policy.pt")
        else:
            no_improve_epochs += 1

        # Early stop.
        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break

    logger.save()
    print(f"Total training time = {(time.time() - start_time):.4f}s")

    # Evaluate model sampling based on fluency, coherence, and diversity.
    test_prompts, test_references = test
    test_prompts = test_prompts[:batch_size]
    test_references = test_references[:batch_size]

    # Generate text w/o policy + metrics.
    start_time = time.time()
    gen_text_wo_policy = []
    for prompt in test_prompts:
        sample, *_ = generate(model, tokenizer, prompt, other_eots)
        gen_text_wo_policy.append(sample)
    print_metrics(gen_text_wo_policy, test_references, w_policy=False)
    print(f"Total gen w/o policy time = {(time.time() - start_time):.4f}s")

    # Generate text w/ policy + metrics.
    gen_text_w_policy = []
    start_time = time.time()
    with torch.no_grad():
        for prompt in test_prompts:
            sample, *_ = generate(model, tokenizer, prompt, other_eots, policy)
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
