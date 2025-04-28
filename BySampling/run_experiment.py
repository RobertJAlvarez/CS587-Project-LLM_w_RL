import argparse
import importlib
import time

import torch  # type: ignore
import tiktoken  # type: ignore
from my_gpt import GPT, Config
from metrics import evaluate_text


def run_experiment(
    prompt: str,
    model_path: str,
    top_k: int,
    temperature: float,
    max_new_tokens: int = 1000,
    use_kv_cache: bool = True,
):
    # 1) Tokenizer.
    enc = tiktoken.get_encoding("cl100k_base")

    # 2) Device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] using device: {device}")

    # 3) Model config & load.
    config = Config(
        vocab_size=enc.n_vocab, n_embd=256, n_head=8, n_layer=4, block_size=128
    )
    model = GPT(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4) Prepare generation functions.
    script_names = ["generate", "generate_AR", "generate_HMC", "generate_MH"]
    methods = {
        name: getattr(importlib.import_module(name), "generate")
        for name in script_names
    }

    # 5) Run each method.
    for name, gen_fn in methods.items():
        print(f"\n=== Running {name} ===")
        start_time = time.time()
        # Generate (may return tokens, or (tokens, time), or text, or (text, time))
        result = gen_fn(
            model=model,
            prompt=prompt,
            tokenizer=enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_kv_cache=use_kv_cache,
        )

        # Unpack result.
        if isinstance(result, tuple) and len(result) == 2:
            output, gen_time = result
        else:
            output = result
            gen_time = time.time() - start_time

        # Determine text.
        if isinstance(output, (list, tuple)) or torch.is_tensor(output):
            token_ids = output.tolist() if torch.is_tensor(output) else list(output)
            text = enc.decode(token_ids)
        elif isinstance(output, str):
            text = output
        else:
            raise ValueError(f"Unrecognized output type from {name}: {type(output)}")

        # Remove end-of-text marker to avoid tokenization errors in metrics.
        text = text.replace("<|endoftext|>", "")

        # Compute metrics on cleaned text.
        metrics = evaluate_text(model, enc, text)
        metrics["generation_time"] = gen_time

        # Print.
        print(f"--- {name} (took {gen_time:.2f}s) ---")
        print(text)
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all four samplers on a single prompt"
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/best_gpt_model.pt")
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    args = parser.parse_args()

    run_experiment(
        prompt=args.prompt,
        model_path=args.model_path,
        top_k=args.top_k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
