import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore

from baseline_sampling import model, tokenizer, device
from data_utils import load_prompts_and_references
from reward_functions import compute_fluency_score, compute_coherence_score
from reward_logger import RewardLogger


class SamplingPolicy(nn.Module):
    """Learn a temperature adjustment based on prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(768, 1)  # 768 = GPT2 hidden size

    def forward(self, prompt_embedding):
        # Temperature in (0, 2)
        return torch.sigmoid(self.linear(prompt_embedding)) * 2.0


policy = SamplingPolicy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-4)


def sample_with_policy(prompt, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        top_p=0.9,
        temperature=temperature.item(),
        top_k=50,
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
                prompt_emb = model.transformer.wte(inputs.input_ids).mean(
                    dim=1
                )  # Mean pooling.

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


if __name__ == "__main__":
    (prompts, references), _ = load_prompts_and_references(train_size=1000)
    print(f"Loaded {len(prompts)} prompt-reference pairs.")

    train_policy(prompts, references, epochs=500)
