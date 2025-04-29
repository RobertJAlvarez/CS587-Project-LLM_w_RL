import torch  # type: ignore
from utils import top_p_filtering


def generate_w_policy(
    model,
    prompt,
    tokenizer,
    policy,
    max_new_tokens=1000,
) -> str:
    """
    Generate text from a prompt using the trained GPT model with KV caching support.

    Args:
        model: The trained GPT model.
        tokenizer: The tokenizer used to encode/decode text.
        prompt: The text prompt to start generation.
        policy: Policy to generate the temperature to be use for the next token.
        max_new_tokens: Maximum number of tokens to generate.

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

    temperature, top_p = policy(model.transformer.wte(tokens))

    # Initialize the past key values to None (no caching yet).
    past_key_values = None
    max_past_key_values_len = model.config.block_size - 1

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
        if next_token.item() == tokenizer.eot_token:
            break
        else:
            # Append the token to our sequence.
            tokens = torch.cat((tokens, next_token), dim=1)

        # Update temperature and top_p.
        temperature, top_p = policy(model.transformer.wte(tokens))

    # Decode the tokens.
    generated_text = tokenizer.decode(tokens[0].tolist())

    # Return both the generated text and timing information.
    return generated_text, temperature, top_p
