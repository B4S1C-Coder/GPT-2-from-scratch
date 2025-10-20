import torch
from classification.spam_model import construct_classification_model, classify_review
from impl.utils import perform_non_cpu_backend_check, get_gpt2_tokenizer

device = perform_non_cpu_backend_check()

# Load weights
model = construct_classification_model(device=device, num_classes=2)
model.load_state_dict(torch.load('bin/gpt2_spam_classif_weights.pth', map_location=device))

predicted_label = classify_review(
    text="You are a winner you have been specially selected to receive $1000 reward.",
    model=model,
    tokenizer=get_gpt2_tokenizer(),
    max_length=120,
    device=device
)

print(f"Prediction: {predicted_label}")