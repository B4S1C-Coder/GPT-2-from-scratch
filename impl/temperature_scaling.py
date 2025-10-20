import torch
import matplotlib.pyplot as plt
from datetime import datetime

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

def illustration():
    vocab = {
        "closer":  0,
        "every":   1,
        "effort":  2,
        "forward": 3,
        "inches":  4,
        "moves":   5,
        "pizza":   6,
        "toward":  7,
        "you":     8,
    }

    inverse_vocab = { v: k for k, v in vocab.items() }
    # Let the input is "Every effort moves you", let we get the logits tensor as:
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )
    probas = torch.softmax(next_token_logits, dim=0)
    print(probas)

    # Earlier we used to do greedy decoding:
    next_token_id = torch.argmax(probas).item()
    print(f"Greedy Decoding Predicition: {next_token_id} <-> {inverse_vocab[int(next_token_id)]}")

    # Now instead of greedy decoding, we will use temperature scaling via the Multinomial Distribution
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print(f"MultinomialDist Predicition: {next_token_id} <-> {inverse_vocab[int(next_token_id)]}")

    tmeperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in tmeperatures]
    print(f"Scaled probablities:", scaled_probas)

    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(tmeperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')

    ax.set_label('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"temperature_scaling_{datetime.now().isoformat().replace(':', '-')}.png")


if __name__ == "__main__":
    illustration()