import torch  # type: ignore
import tiktoken  # type: ignore
import argparse
import time
from my_gpt import GPT, Config


def generate_HMC(
    model,
    prompt,
    tokenizer,
    max_new_tokens=1000,
    temperature=1.0,
    top_k=40,
    use_kv_cache=True,
    hmc_steps=10,  # number of leapfrog steps
    hmc_step_size=0.005,  # leapfrog step size
):
    """
    Generate text from a prompt using the trained GPT model with KV caching support
    and an HMC-based sampler for each next-token distribution.
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
            if past_key_values[0][0].size(2) > max_past:
                past_key_values = tuple(
                    (k[:, :, -max_past:], v[:, :, -max_past:])
                    for k, v in past_key_values
                )

        # 2) Forward pass to get logits
        with torch.no_grad():
            logits, new_past = model(
                context, past_key_values=past_key_values, use_cache=use_kv_cache
            )
            if use_kv_cache:
                past_key_values = new_past

        # 3) Extract and temperature-scale the last-token logits
        next_logits = logits[:, -1, :] / temperature  # (1, V)
        #    apply top-k filtering
        if top_k > 0:
            topk_vals, _ = next_logits.topk(top_k, dim=-1)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            next_logits = next_logits.masked_fill(
                next_logits < threshold, -float("Inf")
            )

        # 4) Build initial probability vector q₀
        q0 = next_logits.softmax(dim=-1)  # (1, V)

        # 5) HMC on the simplex
        # Define energy U(q) = -q·logits + Σ qᵢ log qᵢ
        def U(q):
            return -(q * next_logits).sum(dim=-1) + (q * torch.log(q)).sum(dim=-1)

        # initialize continuous state and momentum
        q = q0.clone().detach().requires_grad_(True)
        p_mom = torch.randn_like(q)

        # half-step momentum update
        (grad_q,) = torch.autograd.grad(U(q), q)
        p_mom = p_mom - 0.5 * hmc_step_size * grad_q

        # full leapfrog steps
        for _ in range(hmc_steps):
            q = q + hmc_step_size * p_mom
            q = torch.clamp(q, min=1e-6)  # stay in simplex interior
            (grad_q,) = torch.autograd.grad(U(q), q)
            p_mom = p_mom - hmc_step_size * grad_q

        # final half-step
        (grad_q,) = torch.autograd.grad(U(q), q)
        p_mom = p_mom - 0.5 * hmc_step_size * grad_q

        # 6) Metropolis acceptance
        def H(q, p):
            return U(q) + 0.5 * (p**2).sum(dim=-1)

        current_H = H(q0, torch.zeros_like(q0))
        proposed_H = H(q, p_mom)
        if torch.rand(()) < torch.exp(current_H - proposed_H):
            q_star = q.detach()
        else:
            q_star = q0

        # 7) Discretize: draw next token from q_star
        probs = q_star / q_star.sum(dim=-1, keepdim=True)
        next_token = probs.multinomial(num_samples=1)  # (1,1)

        # Append and check for end-of-text
        if next_token.item() == tokenizer.eot_token:
            tokens = torch.cat([tokens, next_token], dim=1)
            break
        tokens = torch.cat([tokens, next_token], dim=1)
        # ————————————————————————————————————————————————————————

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

    generated_text, generation_time = generate_HMC(
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
