import torch
import matplotlib.pyplot as plt
from datetime import datetime
from impl.gelu_activation import GELU

torch.set_default_device("cuda")

class ExampleDeepNeuralNetwork(torch.nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut

        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]), GELU())
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)

            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        
        return x

def print_gradients(model: torch.nn.Module, x) -> list[float]:
    output = model(x)
    target = torch.tensor([[0.]])

    loss = torch.nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    grads = []

    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            item = param.grad.abs().mean().item()
            print(f"{name} has gradient mean of {item}")

            grads.append(item)

    return grads

def plot_grads_v1(without_shortcut: list[float], with_shortcut: list[float]) -> str:
    assert len(with_shortcut) == len(without_shortcut), "Lengths of both sets must be same."

    layers = [i for i in range(len(with_shortcut))]
    plt.plot(layers[::-1], without_shortcut[::-1], label="Without shortcut connections")
    plt.plot(layers[::-1], with_shortcut[::-1], label="With Shortcut connections")

    plt.legend(loc='upper right')
    output_path = f"{datetime.now().isoformat().replace(':', '-')}.png"
    plt.savefig(output_path)

    return output_path

def plot_grads(without_shortcut: list[float], with_shortcut: list[float]) -> str:
    assert len(with_shortcut) == len(without_shortcut), "Lengths of both sets must be same."

    num_layers = len(without_shortcut)
    layers = list(range(num_layers))  # 0 = last layer, num_layers-1 = first layer

    # Reverse gradients so that "last layer" comes first
    without_shortcut = without_shortcut[::-1]
    with_shortcut = with_shortcut[::-1]

    plt.figure(figsize=(8, 5))
    plt.semilogy(layers, without_shortcut, marker="o", label="Without shortcut connections")
    plt.semilogy(layers, with_shortcut, marker="o", label="With shortcut connections")

    plt.xlabel("Layer index (0 = output layer, left → right = deeper layers)")
    plt.ylabel("Mean Gradient Magnitude (log scale)")
    plt.title("Effect of Shortcut Connections on Gradient Flow")
    plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.legend(loc='upper right')

    output_path = f"{datetime.now().isoformat().replace(':', '-')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    sample_input = torch.tensor([[1., 0., -1., 0., 0., 0., 0., 0., 0., 0.]], device="cuda")


    dnn_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes=[10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 1], use_shortcut=False
    )

    without_shortcut_grads = print_gradients(dnn_without_shortcut, sample_input)

    print("----------------------------")

    dnn_with_shortcut_conns = ExampleDeepNeuralNetwork(
        layer_sizes=[10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 1], use_shortcut=True
    )

    with_shortcut_grads = print_gradients(dnn_with_shortcut_conns, sample_input)

    saved_path = plot_grads(without_shortcut_grads, with_shortcut_grads)
    print(f"Output saved to: {saved_path}")

if __name__ == "__main__":
    main()