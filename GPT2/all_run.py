from datasets import load_dataset  # type: ignore
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
import json


def generate_text(
    prompt,
    max_length: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def compute_fluency_score(text):
    """Simple length-based fluency (better: use a small language model later)."""
    words = nltk.word_tokenize(text)
    if len(words) < 5:
        return -1.0  # Penalize very short text.
    return 1.0


def compute_coherence_score(reference, generated):
    """Use BLEU score for rough coherence."""
    reference_tokens = nltk.word_tokenize(reference)
    generated_tokens = nltk.word_tokenize(generated)
    return sentence_bleu([reference_tokens], generated_tokens)


def compute_diversity_score(samples) -> float:
    """Self-BLEU — lower means more diverse."""
    scores = []
    for i, sample in enumerate(samples):
        others = samples[:i] + samples[i + 1 :]
        scores.extend(
            [
                sentence_bleu([nltk.word_tokenize(other)], nltk.word_tokenize(sample))
                for other in others
            ]
        )
    return 1.0 - sum(scores) / len(scores)


class RewardLogger:
    def __init__(self, save_path="rewards.json") -> None:
        self.rewards = []
        self.save_path = save_path

    def log(self, reward_value) -> None:
        self.rewards.append(reward_value)

    def save(self) -> None:
        with open(self.save_path, "w") as f:
            json.dump(self.rewards, f)

    def plot(self) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards, label="Average Reward per Epoch", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.title("Reward Convergence Over Time")
        plt.legend()
        plt.grid()
        plt.show()


class SamplingPolicy(nn.Module):
    """Learn a temperature adjustment based on prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(768, 1)  # 768 = GPT2 hidden size.

    def forward(self, prompt_embedding):
        # Temperature in (0, 2).
        return torch.sigmoid(self.linear(prompt_embedding)) * 2.0


def sample_with_policy(prompt, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        # max_length=50,
        do_sample=True,
        top_p=0.9,
        temperature=temperature.item(),
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def train_policy(prompts, references, epochs=1000):
    model.eval()
    logger = RewardLogger()

    for epoch in range(epochs):
        optimizer.zero_grad()
        rewards = []
        log_probs = []

        for prompt, reference in zip(prompts, references):
            # Get prompt embedding.
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                # Mean pooling.
                prompt_emb = model.transformer.wte(inputs.input_ids).mean(dim=1)

            # Get temperature.
            temperature = policy(prompt_emb)

            # Sample.
            sample = sample_with_policy(prompt, temperature)

            # Compute reward.
            fluency = compute_fluency_score(sample)
            coherence = compute_coherence_score(reference, sample)
            reward = fluency + coherence  # This can be weighted.

            # Log probability (assume log_prob ~ -temperature for simple start).
            log_prob = torch.log(temperature + 1e-8)

            rewards.append(reward)
            log_probs.append(log_prob)

        rewards = torch.tensor(rewards, device=device)
        log_probs = torch.stack(log_probs)

        # Policy gradient loss: maximize expected reward.
        loss = -(rewards * log_probs).mean()

        loss.backward()
        optimizer.step()

        avg_reward = rewards.mean().item()
        logger.log(avg_reward)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f} Reward {avg_reward:.4f}")

    logger.save()
    logger.plot()


def load_prompts_and_references(train_pct: int = 10, test_pct: str = 2):
    """
    Load prompts and references from OpenWebText.
    Split into training and test sets.
    """
    tot_pct = train_pct + test_pct
    assert tot_pct <= 100, "Train + test percent > 100"

    print("Loading datasets...")
    # openwebtext = load_dataset(
    #     "openwebtext", split=f"train[:{tot_pct}%]", trust_remote_code=True
    # )
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


def evaluate_sampling(prompts, references):
    """Evaluate model sampling based on fluency, coherence, and diversity."""
    generated_texts = []

    for prompt in prompts:
        generated_texts.append(generate_text(prompt))

    # Metrics.
    fluency_scores = [compute_fluency_score(g) for g in generated_texts]
    coherence_scores = [
        compute_coherence_score(r, g) for r, g in zip(references, generated_texts)
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
        print(f"\nPrompt: {prompts[i]}")
        print(f"Generated: {generated_texts[i]}")
        print(f"Reference: {references[i]}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
policy = SamplingPolicy().to(device)

nltk.download("punkt")
nltk.download("punkt_tab")

if __name__ == "__main__":
    # Training.
    train, test = load_prompts_and_references()
    train_prompts, train_references = train

    print(f"Loaded {len(train_prompts)} prompt-reference pairs.")
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    train_policy(train_prompts, train_references, epochs=500)

    # Evaluation.
    test_prompts, test_references = test
    evaluate_sampling(test_prompts, test_references)
