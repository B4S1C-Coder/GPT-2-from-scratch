import torch

def illustration():
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)

    print("Top logits   :", top_logits)
    print("Top positions:", top_pos)

    new_logits = torch.where(
        condition=next_token_logits < top_logits[-1],
        input=torch.tensor(float("-inf")),
        other=next_token_logits
    )

    print(new_logits)
    topk_probas = torch.softmax(new_logits, dim=0)
    print(topk_probas)

if __name__ == "__main__":
    illustration()