import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datasets import get_mnist_loaders, get_cifar_loaders
from utils import (
    plot_training_history,
    count_parameters,
    save_model,
    compare_models
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001


# === МОДЕЛИ ===
class FullyConnectedMNIST(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 if input_channels == 1 else 32 * 32 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(x)


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))


class ResNetLikeCNN(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.resblock1 = ResidualBlock(64)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.resblock1(x)
        x = self.pool(x)
        return self.fc(x)


# === ОБУЧЕНИЕ ===
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(1) == y).sum().item()
    return total_loss / len(train_loader), correct / len(train_loader.dataset)


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            correct += (output.argmax(1) == y).sum().item()
            all_preds.extend(output.argmax(1).cpu())
            all_targets.extend(y.cpu())
    return total_loss / len(test_loader), correct / len(test_loader.dataset), all_preds, all_targets


def plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {name}")
    plt.savefig(f"{name}_conf_matrix.png")
    plt.clf()


# === ЭКСПЕРИМЕНТ ===
def run_experiment(model_class, dataset='mnist', model_name='model', save_path=None):
    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)
        input_channels = 1
    else:
        train_loader, test_loader = get_cifar_loaders(BATCH_SIZE)
        input_channels = 3

    model = model_class(input_channels=input_channels).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    history = {'train_losses': [], 'test_losses': [], 'train_accs': [], 'test_accs': []}

    start_time = time.time()
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)

        history['train_losses'].append(train_loss)
        history['test_losses'].append(test_loss)
        history['train_accs'].append(train_acc)
        history['test_accs'].append(test_acc)

        print(f"[{model_name}] Epoch {epoch + 1}/{EPOCHS}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

    total_time = time.time() - start_time
    print(f"{model_name} training time: {total_time:.2f}s")
    print(f"{model_name} parameters: {count_parameters(model):,}")

    _, _, preds, targets = evaluate(model, test_loader, criterion)
    plot_training_history(history)
    plot_confusion(targets, preds, model_name)

    if save_path:
        save_model(model, save_path)

    return history

# === MAIN ===
if __name__ == "__main__":
    print("===== CIFAR-10 EXPERIMENTS =====")
    fc_cifar = run_experiment(FullyConnectedMNIST, 'cifar', 'FC_CIFAR')
    res_cifar = run_experiment(ResNetLikeCNN, 'cifar', 'ResCNN_CIFAR')