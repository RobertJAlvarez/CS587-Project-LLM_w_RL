import argparse
import tiktoken  # type: ignore
import os
import torch  # type: ignore
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt  # type: ignore
from functools import partial

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
from Juanes_metric import compute_perplexity, distinct_n, repetition_rate


def test_models(num_samples: int = 32) -> None:
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
        model.load_state_dict(
            torch.load("models/best_gpt_model.pt"), map_location=device
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    model.to(device)
    model.eval()  # Set the model to evaluation mode.

    # Initialize policy.
    policy = SamplingPolicy().to(device)

    # Load the trained weights.
    try:
        policy.load_state_dict(torch.load("models/best_policy.pt"), map_location=device)
    except Exception as e:
        print(f"Error loading policy: {e}")
        exit(1)
    policy.to(device)
    policy.eval()  # Set the policy to evaluation mode.

    # Generate text using specified setting.
    print("All models use 'KV cache'")

    # Initialize storage.
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Evaluate each model sampling based on fluency, coherence, and diversity.
    gen_fs = [
        partial(generate, model=model, tokenizer=tokenizer),
        partial(generate_w_policy, model=model, tokenizer=tokenizer, policy=policy),
        partial(generate_AR, model=model, tokenizer=tokenizer),
        partial(generate_HMC, model=model, tokenizer=tokenizer),
        partial(generate_MH, model=model, tokenizer=tokenizer),
    ]
    titles = [
        "Original model",
        "Model w/ RL",
        "Model w/ AR",
        "Model w/ HMC",
        "Model w/ MH",
    ]

    start_time_loop = time.time()
    for k in [4, 8, 12, 16, 20]:
        for in_dist, d_name in zip(
            [True, False], ["in_distribution", "out_of_distribution"]
        ):
            # Load prompts/references.
            if in_dist:
                prompts, references = sample_paragraph_splits(
                    num_samples=num_samples, k=k
                )
            else:
                prompts, references = load_prompts_and_references(
                    num_samples=num_samples, k=k
                )

            for gen_f, title in zip(gen_fs, titles):
                # Generate text per prompt.
                gen_texts = []
                start_time = time.time()
                for prompt in prompts:
                    sample, *_ = gen_f(prompt=prompt)
                    gen_texts.append(sample)

                # Compute elapse time.
                results[d_name][title]["time"].append(time.time() - start_time)

                # Compute metrics.
                fluency_scores = [compute_fluency_score(g) for g in gen_texts]
                coherence_scores = [
                    compute_coherence_score(r, g) for r, g in zip(references, gen_texts)
                ]
                diversity_score = compute_diversity_score(gen_texts)
                rouge_l_scores = [
                    compute_rouge_l(r, g) for r, g in zip(references, gen_texts)
                ]

                # Juanes metrics.
                perplexity_scores = [
                    compute_perplexity(model, tokenizer, g) for g in gen_texts
                ]
                list_words = [g.split() for g in gen_texts]
                d1s = [distinct_n(words, 1) for words in list_words]
                d2s = [distinct_n(words, 2) for words in list_words]
                repetition_rates = [repetition_rate(words) for words in list_words]

                # Store mean scores.
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

                results[d_name][title]["perplexity"].append(
                    sum(perplexity_scores) / len(perplexity_scores)
                )
                results[d_name][title]["diversity1"].append(sum(d1s) / len(d1s))
                results[d_name][title]["diversity2"].append(sum(d2s) / len(d2s))
                results[d_name][title]["repetition_rates"].append(
                    sum(repetition_rates) / len(repetition_rates)
                )

                # Print example at random.
                print(
                    f"=========== {title}, k={k}, in_dist={in_dist} START ==========="
                )
                print(f"** Prompt **\n{prompts[0]}")
                print(f"** Gen text **\n{gen_texts[0]}")
                print(f"** Reference **\n{references[0]}")
                print(f"=========== {title} END ===========")

    total_time_loop = time.time() - start_time_loop
    print(f"Total time for {num_samples} samples: {total_time_loop:.2f}")

    dir_name = f"plots-{num_samples}_samples"
    os.makedirs(dir_name, exist_ok=True)

    metrics = [
        "fluency",
        "coherence",
        "diversity",
        "rouge_l",
        "perplexity",
        "diversity1",
        "diversity2",
        "repetition_rates",
    ]
    colors = ["blue", "green", "red", "purple", "orange"]

    for d_name, dataset_results in results.items():
        for metric in metrics:
            plt.figure(figsize=(8, 6))
            for title, color in zip(titles, colors):
                times = dataset_results[title]["time"]

                # Compute average time across k for that model.
                avg_time = sum(times) / len(times)

                plt.plot(
                    dataset_results[title]["k"],
                    dataset_results[title][metric],
                    label=f"{title} (avg {avg_time:.2f}s)",
                    marker="o",
                    color=color,
                )
            plt.title(f"{metric.capitalize()} vs k ({d_name})")
            plt.xlabel("k")
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{dir_name}/{metric}_vs_k_{d_name}.png")
            plt.close()

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with a trained RoPE GPT model"
    )
    parser.add_argument(
        "--num_samples", type=int, default=32, help="Number of paragraph samples"
    )
    args = parser.parse_args()

    test_models(num_samples=args.num_samples)
