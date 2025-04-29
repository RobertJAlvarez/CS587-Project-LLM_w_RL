import argparse
import tiktoken  # type: ignore
import time
import torch  # type: ignore

from my_gpt import GPT, Config


def generate_AR(
    model,
    prompt,
    tokenizer,
    max_new_tokens=1000,
    temperature=1.0,
    top_k=40,
    use_kv_cache=True,
):
    """
    Generate text from a prompt using the trained GPT model with KV caching support.
    """
    model.eval()
    # Encode the prompt and move to the model’s device
    encoded_prompt = tokenizer.encode(prompt)
    tokens = torch.tensor(
        encoded_prompt, dtype=torch.long, device=model.lm_head.weight.device
    ).unsqueeze(0)

    start_time = time.time()
    past_key_values = None
    max_past = model.config.block_size - 1

    for _ in range(max_new_tokens):
        # 1) Build the input chunk
        if not use_kv_cache or past_key_values is None:
            context = tokens[:, -model.config.block_size :]
        else:
            context = tokens[:, -1:]
            # Trim old KV entries if they exceed the window
            if past_key_values[0][0].size(2) > max_past:
                past_key_values = tuple(
                    (k_cache[:, :, -max_past:], v_cache[:, :, -max_past:])
                    for k_cache, v_cache in past_key_values
                )

        # 2) Forward pass
        with torch.no_grad():
            logits, new_past = model(
                context, past_key_values=past_key_values, use_cache=use_kv_cache
            )
            if use_kv_cache:
                past_key_values = new_past

        # 3) Isolate last token logits & apply temperature
        next_logits = logits[:, -1, :] / temperature  # shape (1, V)

        # 4) Top‑k filtering
        if top_k > 0:
            # torch.topk returns (values, indices)
            topk_vals, _ = next_logits.topk(top_k, dim=-1)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            next_logits = next_logits.masked_fill(
                next_logits < threshold, -float("Inf")
            )

        # 5) Softmax via tensor method
        probs = next_logits.softmax(dim=-1)  # shape (1, V)

        # 6) Sample next token
        next_token = probs.multinomial(num_samples=1)  # shape (1, 1)

        while True:
            # propose a token
            proposal = probs.multinomial(num_samples=1)  # shape (1,1)
            p = probs[
                0, proposal.item()
            ].item()  # its model probability accept with probability = p
            if torch.rand(1).item() < p:
                next_token = proposal
                break
            # else: rej

        # 7) Append and check stop
        if next_token.item() == tokenizer.eot_token:
            # tokens = torch.cat([tokens, next_token], dim=1)
            break
        tokens = torch.cat([tokens, next_token], dim=1)

    generation_time = time.time() - start_time
    generated_text = tokenizer.decode(tokens[0].tolist())
    return generated_text, generation_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with a trained RoPE GPT model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The ship was",
        help="Text prompt to start generation",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_gpt_model.pt",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_k", type=int, default=40, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--use_kv_cache", action="store_true", help="Use KV caching for generation"
    )
    args = parser.parse_args()

    # Load the tokenizer (GPT-4 tokenizer)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model configuration (must match the trained model's configuration)
    config = Config(
        vocab_size=tokenizer.n_vocab, n_embd=256, n_head=8, n_layer=4, block_size=128
    )

    # Initialize the model
    model = GPT(config)

    # Load the trained weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Move model to the appropriate device
    model.to(device)

    # Generate text using specified setting
    print(
        f"\nGenerating text using {'KV cache' if args.use_kv_cache else 'standard generation'}..."
    )

    generated_text, generation_time = generate_AR(
        model=model,
        prompt=args.prompt,
        tokenizer=tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_kv_cache=args.use_kv_cache,
    )

    # Print timing information
    print(f"Generation completed in {generation_time:.4f} seconds")

    # Print the generated text
    print("\nGenerated Text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
