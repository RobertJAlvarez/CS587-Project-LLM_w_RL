import argparse
import tiktoken  # type: ignore
import os
import torch  # type: ignore
import random
from collections import defaultdict
import matplotlib.pyplot as plt  # type: ignore

from my_gpt import GPT, Config
from load_dataset import sample_paragraph_splits, load_prompts_and_references
from sampling_policy import SamplingPolicy

from generate import generate
from generate_w_policy import generate_w_policy
from generate_AR import generate_AR
from generate_HMC import generate_HMC
from generate_MH import generate_MH

from metrics import (
    compute_fluency_score,
    compute_coherence_score,
    compute_diversity_score,
    compute_rouge_l,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with a trained RoPE GPT model"
    )
    parser.add_argument(
        "--num_samples", type=int, default=128, help="Number of paragraph samples"
    )
    args = parser.parse_args()

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

    # Initialize policy.
    policy = SamplingPolicy().to(device)

    # Generate text using specified setting.
    print("All models use 'KV cache'")

    # Initialize storage.
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Evaluate each model sampling based on fluency, coherence, and diversity.
    gen_fs = [generate, generate_w_policy, generate_AR, generate_HMC, generate_MH]
    titles = [
        "Original model",
        "Model w/ RL",
        "Model with AR",
        "Model with HMC",
        "Model with MH",
    ]

    for k in [4, 8, 12, 16, 20]:
        for in_dist, d_name in zip(
            [True, False], ["in_distribution", "out_of_distribution"]
        ):
            # Load prompts/references.
            if in_dist:
                prompts, references = sample_paragraph_splits(
                    num_samples=args.num_samples, k=k
                )
            else:
                prompts, references = load_prompts_and_references(
                    num_samples=args.num_samples, k=k
                )

            for gen_f, title in zip(gen_fs, titles):
                # Generate text per prompt.
                gen_texts = []
                for prompt in prompts:
                    if gen_fs == generate_w_policy:
                        sample, *_ = gen_f(model, prompt, tokenizer, policy)
                    else:
                        sample, *_ = gen_f(model, prompt, tokenizer)
                    gen_texts.append(sample)

                # Compute metrics.
                fluency_scores = [compute_fluency_score(g) for g in gen_texts]
                coherence_scores = [
                    compute_coherence_score(r, g) for r, g in zip(references, gen_texts)
                ]
                diversity_score = compute_diversity_score(gen_texts)
                rouge_l_scores = [
                    compute_rouge_l(r, g) for r, g in zip(references, gen_texts)
                ]

                # Store mean scores
                results[d_name][title]["k"].append(k)
                results[d_name][title]["fluency"].append(
                    sum(fluency_scores) / len(fluency_scores)
                )
                results[d_name][title]["coherence"].append(
                    sum(coherence_scores) / len(coherence_scores)
                )
                results[d_name][title]["diversity"].append(diversity_score)
                results[d_name][title]["rouge_l"].append(
                    sum(rouge_l_scores) / len(rouge_l_scores)
                )

                # Print 2 example at random.
                for i in range(2):
                    idx = random.randint(0, len(gen_texts))
                    print(f"=========== {title} START ===========")
                    print(f"** Prompt **\n{prompt[idx]}")
                    print(f"** Gen text **\n{gen_texts[idx]}")
                    print(f"** Reference **\n{references[idx]}")
                    print(f"=========== {title} END ===========")

    os.makedirs("plots", exist_ok=True)

    metrics = ["fluency", "coherence", "diversity", "rouge_l"]
    colors = ["blue", "green", "red", "purple", "orange"]

    for dataset_name, dataset_results in results.items():
        for metric in metrics:
            plt.figure(figsize=(8, 6))
            for title, color in zip(titles, colors):
                plt.plot(
                    dataset_results[title]["k"],
                    dataset_results[title][metric],
                    label=title,
                    marker="o",
                    color=color,
                )
            plt.title(f"{metric.capitalize()} vs k ({dataset_name})")
            plt.xlabel("k")
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.savefig(f"plots/{metric}_vs_k_{dataset_name}.png")
            plt.close()
