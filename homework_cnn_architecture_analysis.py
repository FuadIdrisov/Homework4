import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from datasets import get_mnist_loaders
from utils import plot_training_history, count_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === МОДЕЛИ С РАЗНЫМИ ЯДРАМИ СВЕРТКИ ===
class ConvKernelCNN(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class MixedKernelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Conv2d(1, 16, kernel_size=1)
        self.conv3x3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = F.relu(self.conv1x1(x))
        x = self.pool(F.relu(self.conv3x3(x)))
        return self.fc(x)


# === МОДЕЛИ С РАЗНОЙ ГЛУБИНОЙ ===
class DeepCNN(nn.Module):
    def __init__(self, depth=2, residual=False):
        super().__init__()
        layers = []
        in_channels = 1
        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels, 32, 3, padding=1))
            layers.append(nn.ReLU())
            in_channels = 32
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 10)
        )
        self.residual = residual

    def forward(self, x):
        if self.residual:
            residual = x
            for layer in self.conv:
                x = layer(x)
            x = x + F.interpolate(residual, size=x.shape[-2:])
        else:
            x = self.conv(x)
        x = self.pool(x)
        return self.fc(x)


# === ОБУЧЕНИЕ ===
def train_eval(model, train_loader, test_loader, name):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    history = {'train_losses': [], 'test_losses': [], 'train_accs': [], 'test_accs': []}
    start = time.time()
    for epoch in range(5):
        model.train()
        total_loss, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item()
                correct += (out.argmax(1) == y).sum().item()
        test_loss = total_loss / len(test_loader)
        test_acc = correct / len(test_loader.dataset)

        history['train_losses'].append(train_loss)
        history['test_losses'].append(test_loss)
        history['train_accs'].append(train_acc)
        history['test_accs'].append(test_acc)

        print(f"[{name}] Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

    print(f"{name} Params: {count_parameters(model)}")
    print(f"{name} Time: {time.time() - start:.2f}s")
    plot_training_history(history)

    return model


# === ВИЗУАЛИЗАЦИЯ АКТИВАЦИЙ ===
def visualize_activations(model, loader, name):
    model.eval()
    for x, _ in loader:
        x = x.to(DEVICE)
        with torch.no_grad():
            # Найдём первый Conv2d слой и применим его
            first_conv = None
            for layer in model.modules():
                if isinstance(layer, nn.Conv2d):
                    first_conv = layer
                    break
            if first_conv is None:
                print("❌ Conv2d слой не найден.")
                return
            act = first_conv(x)

        act = act[0].cpu() 
        fig, axes = plt.subplots(1, min(8, act.shape[0]), figsize=(16, 2))
        for i in range(min(8, act.shape[0])):
            axes[i].imshow(act[i], cmap='gray')
            axes[i].axis('off')
        plt.suptitle(f"{name} - First 8 Feature Maps")
        plt.show()
        break


if __name__ == '__main__':
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    print("=== Kernel Size Analysis ===")
    for k in [3, 5, 7]:
        model = ConvKernelCNN(kernel_size=k)
        trained = train_eval(model, train_loader, test_loader, f"Kernel{k}x{k}")
        visualize_activations(trained, test_loader, f"Kernel{k}x{k}")

    mixed = MixedKernelCNN()
    trained_mixed = train_eval(mixed, train_loader, test_loader, "Mixed(1x1+3x3)")
    visualize_activations(trained_mixed, test_loader, "Mixed")

    print("=== Depth Analysis ===")
    for depth in [2, 4, 6]:
        model = DeepCNN(depth=depth)
        trained = train_eval(model, train_loader, test_loader, f"Depth{depth}")
        visualize_activations(trained, test_loader, f"Depth{depth}")

    resnet = DeepCNN(depth=6, residual=True)
    trained_res = train_eval(resnet, train_loader, test_loader, "ResidualCNN")
    visualize_activations(trained_res, test_loader, "Residual")
