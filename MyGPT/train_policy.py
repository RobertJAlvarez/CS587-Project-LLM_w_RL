import argparse
from datasets import load_dataset  # type: ignore
import json
import nltk  # type: ignore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
import random
import tiktoken  # type: ignore
import time
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore

from my_gpt import GPT, Config


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=1000,
    temperature=0.5,
    top_k=10,
    use_kv_cache=True,
):
    """
    Generate text from a prompt using the trained GPT model with KV caching support.

    Args:
        model: The trained GPT model.
        tokenizer: The tokenizer used to encode/decode text.
        prompt: The text prompt to start generation.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Controls randomness (higher = more random).
        top_k: Number of highest probability tokens to consider for sampling.
        use_kv_cache: Whether to use KV caching for more efficient generation.

    Returns:
        The generated text including the prompt and generation time.
    """
    model.eval()  # Set the model to evaluation mode.

    # Encode the prompt.
    encoded_prompt = tokenizer.encode(prompt)
    tokens = (
        torch.tensor(encoded_prompt, dtype=torch.long)
        .unsqueeze(0)
        .to(model.lm_head.weight.device)
    )

    # Track timing for performance analysis.
    start_time = time.time()

    # Initialize the past key values to None (no caching yet).
    past_key_values = None
    max_past_key_values_len = model.config.block_size - 1

    # Generate tokens one at a time.
    for _ in range(max_new_tokens):
        # For KV cache: after first iteration, only process the last token.
        # For no KV cache: always process full sequence within block size limit.
        if not use_kv_cache or past_key_values is None:
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
            if use_kv_cache:
                past_key_values = new_past_key_values

        # Focus on the last token's predictions.
        logits = logits[:, -1, :] / temperature

        # Apply top-k filtering.
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            # Set other logits outside top-l to a value of -inf.
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Apply softmax to get probabilities.
        probs = torch.softmax(logits, dim=-1)

        # Sample from the distribution.
        next_token = torch.multinomial(probs, num_samples=1)

        # If we reach the end of text token, stop.
        if next_token.item() == tokenizer.eot_token:
            break
        else:
            # Append the token to our sequence.
            tokens = torch.cat((tokens, next_token), dim=1)

    # Calculate generation time.
    generation_time = time.time() - start_time

    # Decode the tokens.
    generated_text = tokenizer.decode(tokens[0].tolist())

    # Return both the generated text and timing information.
    return generated_text, generation_time


def compute_fluency_score(text):
    """
    A simple proxy for fluency: longer sentences, proper punctuation.
    (Optional improvement: use perplexity from a small language model.)
    """
    sentences = nltk.tokenize.sent_tokenize(text)
    if not sentences:
        return 0.0

    avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(
        sentences
    )

    # Normalize to a range, e.g., divide by 20 (typical sentence length).
    return min(avg_sentence_length / 20.0, 1.0)


def compute_coherence_score(reference, generated):
    """
    Use BLEU score between reference and generated text.
    """
    reference_tokens = nltk.tokenize.word_tokenize(reference.lower())
    generated_tokens = nltk.tokenize.word_tokenize(generated.lower())
    return sentence_bleu(
        [reference_tokens],
        generated_tokens,
        smoothing_function=SmoothingFunction().method4,
    )


def compute_diversity_score(samples) -> float:
    """
    Compute Self-BLEU: lower is better (more diversity).
    """
    n = len(samples)

    # Perfect diversity if only one sample
    if n <= 1:
        return 1.0

    tokenized_samples = [
        nltk.tokenize.word_tokenize(sample.lower()) for sample in samples
    ]
    total_self_bleu = sum(
        [
            sentence_bleu(
                tokenized_samples[:i] + tokenized_samples[i + 1 :],
                tokenized_samples[i],
                smoothing_function=SmoothingFunction().method4,
            )
            for i in range(n)
        ]
    )

    return 1.0 - (total_self_bleu / n)


class RewardLogger:
    def __init__(self, save_path="rewards.json") -> None:
        self.rewards = []
        self.save_path = save_path

    def log(self, reward_value) -> None:
        self.rewards.append(reward_value)

    def save(self) -> None:
        with open(self.save_path, "w") as f:
            json.dump(self.rewards, f)


class SamplingPolicy(nn.Module):
    """Learn a temperature adjustment based on prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(256, 1)  # model.config.n_embd = 256

    def forward(self, prompt_embedding):
        # Temperature in (0, 2).
        return torch.sigmoid(self.linear(prompt_embedding)) * 2.0


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
        "--use_kv_cache", action="store_true", help="Use KV caching for generation"
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

    # Download NLTK resources if not already present
    nltk.download("punkt")

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

    # Generate text using specified setting.
    print(
        f"\nTraining policy using {'KV cache' if args.use_kv_cache else 'standard generation'}..."
    )

    # Initialize policy.
    policy = SamplingPolicy().to(device)

    # Load training and testing prompts.
    train, test = load_prompts_and_references()
    train_prompts, train_references = train

    print(f"Loaded {len(train_prompts)} prompt-reference pairs.")
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)  # AdamW w/ lr=5e-4?

    model.eval()
    logger = RewardLogger()

    # TODO: Properly set up the number of epochs and the train prompt per iteration.
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        rewards = []
        log_probs = []
        st = epoch * batch_size
        ed = (epoch + 1) * batch_size
        for prompt, reference in zip(train_prompts[st:ed], train_references[st:ed]):
            # Get prompt embedding.
            encoded_prompt = tokenizer.encode(prompt)
            tokens = (
                torch.tensor(encoded_prompt, dtype=torch.long)
                .unsqueeze(0)
                .to(model.lm_head.weight.device)
            )
            with torch.no_grad():
                # Mean pooling.
                prompt_emb = model.transformer.wte(tokens).mean(dim=1)

            # Get temperature.
            temperature = policy(prompt_emb)

            # Sample.
            sample, _ = generate(
                model, tokenizer, prompt, temperature=temperature.item()
            )

            # Compute reward.
            fluency = compute_fluency_score(sample)
            coherence = compute_coherence_score(reference, sample)

            # Log probability (assume log_prob ~ -temperature for simple start).
            rewards.append(fluency + coherence)  # This can be weighted.
            log_probs.append(torch.log(temperature + 1e-8))

        rewards = torch.tensor(rewards, device=device)
        log_probs = torch.stack(log_probs)

        # Policy gradient loss: maximize expected reward.
        loss = -(rewards * log_probs).mean()

        loss.backward()
        optimizer.step()

        avg_reward = rewards.mean().item()
        logger.log(avg_reward)

        print(f"Epoch {epoch}: Loss {loss.item():.4f} Reward {avg_reward:.4f}")

    logger.save()

    # Evaluate model sampling based on fluency, coherence, and diversity.
    test_prompts, test_references = test

    # Generate text.
    generated_texts = [generate(model, tokenizer, prompt) for prompt in test_prompts]

    # Metrics.
    fluency_scores = [compute_fluency_score(g) for g in generated_texts]
    coherence_scores = [
        compute_coherence_score(r, g) for r, g in zip(test_references, generated_texts)
    ]
    diversity_score = compute_diversity_score(generated_texts)

    print("\n=== Evaluation Results ===")
    print(f"Avg Fluency Score: {sum(fluency_scores)/len(fluency_scores):.4f}")
    print(
        f"Avg Coherence Score (BLEU): {sum(coherence_scores)/len(coherence_scores):.4f}"
    )
    print(f"Diversity Score (1-SelfBLEU): {diversity_score:.4f}")

    # Print some generated examples.
    for i in range(5):
        print(f"\nPrompt: {test_prompts[i]}")
        print(f"Generated: {generated_texts[i]}")
        print(f"Reference: {test_references[i]}")

    # Save policy.
    best_policy_path = "models/best_policy.pt"
    torch.save(model.state_dict(), best_policy_path)

# TODO: Each generation may take a 1 second or more. If 10,000 samples are generated, then it would take at least 3.5 hours.
#   Given that the comparison reference text is only a sentence longer, add "." as an early stop token (in generate()).
#   Then re-run.
