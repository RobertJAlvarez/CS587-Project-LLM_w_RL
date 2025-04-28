import argparse
import tiktoken  # type: ignore
import time
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore

from my_gpt import GPT, Config
from reward_logger import RewardLogger
from utils import top_p_filtering
from metrics import print_metrics, compute_fluency_score, compute_coherence_score
from load_dataset import load_prompts_and_references


def generate(
    model,
    tokenizer,
    prompt,
    other_eots: list = [],
    policy=None,
    max_new_tokens=1000,
    top_p=0.9,
) -> str:
    """
    Generate text from a prompt using the trained GPT model with KV caching support.

    Args:
        model: The trained GPT model.
        tokenizer: The tokenizer used to encode/decode text.
        prompt: The text prompt to start generation.
        max_new_tokens: Maximum number of tokens to generate.
        policy: Policy to generate the temperature to be use for the next token.
        top_p: Consider all token with high probability where its comulative is at least top_p.

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
        prompt_emb = model.transformer.wte(tokens).mean(dim=1)
        temperature = policy(prompt_emb)
    else:
        temperature = 0.5

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
                prompt_emb = model.transformer.wte(tokens).mean(dim=1)
            temperature = policy(prompt_emb)

    # Decode the tokens.
    generated_text = tokenizer.decode(tokens[0].tolist())

    # Return both the generated text and timing information.
    return generated_text


def generate_w_policy(
    model, tokenizer, prompt, other_eots, policy
) -> tuple[str, torch.Tensor]:
    # Sample.
    sample = generate(model, tokenizer, prompt, other_eots, policy)

    encoded_prompt = tokenizer.encode(sample)

    tokens = (
        torch.tensor(encoded_prompt, dtype=torch.long)
        .unsqueeze(0)
        .to(model.lm_head.weight.device)
    )

    # Mean pooling.
    with torch.no_grad():
        prompt_emb = model.transformer.wte(tokens).mean(dim=1)

    # Get temperature.
    return sample, policy(prompt_emb)


class SamplingPolicy(nn.Module):
    """Learn a temperature adjustment based on prompt."""

    def __init__(self) -> None:
        super().__init__()
        # model.config.n_embd = 256
        self.mlp = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, prompt_embedding) -> torch.Tensor:
        # Temperature in (0, 2).
        return torch.sigmoid(self.mlp(prompt_embedding)) * 2.0


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
            sample, temperature = generate_w_policy(
                model, tokenizer, prompt, other_eots, policy
            )

            # Compute reward.
            fluency = compute_fluency_score(sample)
            coherence = compute_coherence_score(reference, sample)

            # Log probability (assume log_prob ~ -temperature for simple start).
            rewards.append(0.6 * fluency + 0.4 * coherence)
            log_probs.append(torch.log(temperature + 1e-8))
            entropies.append(-(temperature * torch.log(temperature + 1e-8)))

        rewards = torch.tensor(rewards, device=device)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        # Policy gradient loss: maximize expected reward (w/ baseline) + mean(entropy).
        loss = ((rewards - rewards.mean()) * log_probs).mean() - 0.01 * entropies.mean()

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
            print(f"New best policy saved with reward {best_avg_reward:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs.")

        # Early stop.
        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break

    logger.save()
    print(f"Total training time = {time.time() - start_time}s")

    # Evaluate model sampling based on fluency, coherence, and diversity.
    test_prompts, test_references = test

    # Generate text w/o policy.
    gen_text_wo_policy = [
        generate(model, tokenizer, prompt, other_eots) for prompt in test_prompts
    ]
    print_metrics(gen_text_wo_policy, test_references, w_policy=False)

    # Generate text w/ policy.
    gen_text_w_policy = []
    with torch.no_grad():
        for prompt in test_prompts:
            sample, _ = generate_w_policy(model, tokenizer, prompt, other_eots, policy)
            gen_text_w_policy.append(sample)

    # Metrics for gen text w/ policy.
    print_metrics(gen_text_w_policy, test_references, w_policy=True)

    # Print some generated examples.
    for i in range(3):
        print("======================")
        print(f"** Prompt **\n{test_prompts[i]}")
        print(f"** Gen w/ policy **\n{gen_text_w_policy[i]}")
        print(f"** Gen w/o policy **\n{gen_text_wo_policy[i]}")
        print(f"** Reference **\n{test_references[i]}")
        print("======================")
