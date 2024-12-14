import torchvision
from torchvision import datasets
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root="./data/", train=True, transform=transform, download=False)
test_data = datasets.MNIST(root="./data/", train=False, transform=transform, download=False)

batch_size = 64
torch.manual_seed(12)

train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10),
            nn.Softmax()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, 784)
        return self.layers(inputs)
    
model = Model()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

for epoch in range(10):
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        predictions = model(inputs)

        loss_ = loss(predictions, labels)
        loss_.backward()
        optimizer.step()

        epoch_loss += loss_.item()

    print(f"Epoch[{epoch+1:02}/10]\t{epoch_loss: .4f}")

num_correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        predictions = model(inputs)
        _, predictions = torch.max(predictions, 1)
        num_correct += (predictions == labels).sum().item()

print(f"Test Accuracy: {100*num_correct/len(test_data): .2f}%")
