import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# 1. Set reproducible seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Determinism (safer reproducibility)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 2. Hyperparameters
batch_size = 64
epochs = 8
learning_rate = 1e-3


# 3. Define CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # after pooling, output is 64x12x12
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# 4. Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 5. Define loss + optimizer
model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

os.makedirs("logs", exist_ok=True)
os.makedirs("data/checkpoints", exist_ok=True)
os.makedirs("logs/misclassified", exist_ok=True)
os.makedirs("python/golden", exist_ok=True)
log_file = "logs/train_fp32.txt"
misclassified_log = "logs/misclassified.txt"

# Run metadata in log (audit trail)
with open(log_file, "a") as f:
    f.write(
        f"seed={seed}, device={device}, torch={torch.__version__}, torchvision={torchvision.__version__}, mean=0.1307, std=0.3081, batch_size={batch_size}, lr={learning_rate}, epochs={epochs}\n"
    )

MISCLASSIFIED_LIMIT = 50
misclassified_count = 0
# 6. Training loop
best_acc = 0
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch}: Train loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
    print(f"Epoch {epoch}: Train loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # 7. Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    misclassified = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += data.size(0)
            # Save misclassified images (cap at MISCLASSIFIED_LIMIT)
            for i in range(data.size(0)):
                if misclassified_count < MISCLASSIFIED_LIMIT and pred[i].item() != target[i].item():
                    img = data[i].cpu().squeeze().numpy()
                    plt.imsave(
                        f"logs/misclassified/epoch{epoch}_batch{batch_idx}_idx{i}_pred{pred[i].item()}_true{target[i].item()}.png",
                        img,
                        cmap="gray",
                    )
                    misclassified.append(
                        f"epoch{epoch}_batch{batch_idx}_idx{i}: pred={pred[i].item()}, true={target[i].item()}"
                    )
                    misclassified_count += 1
    avg_test_loss = test_loss / test_total
    test_accuracy = 100.0 * test_correct / test_total
    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch}: Test loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%\n")
    print(f"Epoch {epoch}: Test loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    if misclassified:
        with open(misclassified_log, "a") as f:
            for line in misclassified:
                f.write(line + "\n")
    if test_accuracy > best_acc:
        best_acc = test_accuracy

# 8. Save checkpoint (richer)
torch.save(
    {
        "state_dict": model.state_dict(),
        "seed": seed,
        "hparams": {
            "batch_size": batch_size,
            "lr": learning_rate,
            "epochs": epochs,
        },
        "best_acc": best_acc,
    },
    "data/checkpoints/mnist_fp32.pt",
)

# Golden outputs for HW tests
with torch.no_grad():
    golden_inputs = test_dataset.data[:32].numpy()
    golden_logits = model(test_dataset.data[:32].unsqueeze(1).float().to(device)).cpu().detach().numpy()
    np.save("python/golden/mnist_inputs.npy", golden_inputs)
    np.save("python/golden/mnist_logits_fp32.npy", golden_logits)

# Print final test accuracy and best_acc
print(f"Final test accuracy: {test_accuracy:.2f}%")
print(f"Best test accuracy: {best_acc:.2f}%")
