from impl.utils import (
    perform_non_cpu_backend_check,
    generate,
    text_to_token_ids,
    token_ids_to_text,
    get_gpt2_tokenizer,
    load_openai_355m_gpt2
)

device = perform_non_cpu_backend_check()

def main():
    tokenizer = get_gpt2_tokenizer()
    model = load_openai_355m_gpt2(device=device, eval_mode=True)

    prompt = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        "'You are a winner you have been specially selected"
        " to receive $1000 cash or a $2000 award.'"
    )

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer),
        max_new_tokens=35,
        context_size=1024,
        top_k=25,
        temperature=1.4
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

if __name__ == "__main__":
    main()
