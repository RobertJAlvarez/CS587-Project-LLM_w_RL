import argparse
import tiktoken  # type: ignore
import time
import torch  # type: ignore

from generate import generate
from generate_w_policy import generate_w_policy
from load_dataset import sample_paragraph_splits, load_prompts_and_references
from metrics import print_metrics
from my_gpt import GPT, Config
from sampling_policy import SamplingPolicy


def test_policy(
    model=None,
    policy=None,
    in_distribution: bool = True,
    k: int = 5,
    num_samples: int = 128,
) -> None:
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
    if not model:
        model = GPT(config)

        # Load the trained weights.
        try:
            model.load_state_dict(
                torch.load("models/best_gpt_model.pt", map_location=device)
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)
    model.eval()  # Set the model to evaluation mode.

    # Generate text using specified setting.
    print("Training policy using 'KV cache'")

    # Load prompts/references.
    if in_distribution:
        prompts, references = sample_paragraph_splits(k=k, num_samples=num_samples)
    else:
        prompts, references = load_prompts_and_references(k=k, num_samples=num_samples)

    print(f"Loaded {len(prompts)} prompt-reference pairs.")

    # Evaluate model sampling based on fluency, coherence, and diversity.

    # Generate text w/o policy + metrics.
    start_time = time.time()
    gen_text_wo_policy = []
    with torch.no_grad():
        for prompt in prompts:
            sample, *_ = generate(model, prompt, tokenizer)
            gen_text_wo_policy.append(sample)
    print_metrics(gen_text_wo_policy, references, w_policy=False)
    print(f"Total gen w/o policy time = {(time.time() - start_time):.4f}s")

    # Generate text w/ policy + metrics.

    # Initialize policy.
    if not policy:
        # Initialize policy.
        policy = SamplingPolicy()
        # Load the trained weights.
        try:
            policy.load_state_dict(
                torch.load("models/best_policy.pt", map_location=device)
            )
        except Exception as e:
            print(f"Error loading policy: {e}")
            exit(1)
    policy.eval()  # Set the policy to evaluation mode.

    gen_text_w_policy = []
    start_time = time.time()
    with torch.no_grad():
        for prompt in prompts:
            sample, *_ = generate_w_policy(model, prompt, tokenizer, policy)
            gen_text_w_policy.append(sample)
    print_metrics(gen_text_w_policy, references, w_policy=True)
    print(f"Total gen w/ policy time = {(time.time() - start_time):.4f}s")

    # Print some generated examples.
    for i in range(3):
        print("======================")
        print(f"** Prompt **\n{prompts[i]}")
        print(f"** Gen w/ policy **\n{gen_text_w_policy[i]}")
        print(f"** Gen w/o policy **\n{gen_text_wo_policy[i]}")
        print(f"** Reference **\n{references[i]}")
        print("======================")


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
        "--k", type=int, default=5, help="First k tokens from paragraph"
    )
    parser.add_argument(
        "--num_samples", type=int, default=128, help="Number of paragraph samples"
    )
    args = parser.parse_args()

    test_policy(
        in_distribution=args.in_distribution, k=args.k, num_samples=args.num_samples
    )
