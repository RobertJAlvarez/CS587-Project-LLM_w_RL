from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch  # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def generate_text(
    prompt,
    top_p: float = 0.9,
    temperature: float = 1.0,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_generate_text() -> None:
    # Load model.
    model.eval()
    print(generate_text("Once upon a time"))


# Example.
if __name__ == "__main__":
    run_generate_text()
