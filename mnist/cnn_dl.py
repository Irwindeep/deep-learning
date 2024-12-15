import sys, os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

import numpy as np
import dl
import dl.nn as nn

train_data, train_labels = np.load("./data/train_images_flat.npy").reshape(60000, 28, 28, 1), np.load("./data/train_labels.npy")
test_data, test_labels = np.load("./data/test_images_flat.npy").reshape(10000, 28, 28, 1), np.load("./data/test_labels.npy")

train_data, train_labels = dl.Tensor(train_data), dl.Tensor(train_labels)
test_data, test_labels = dl.Tensor(test_data), dl.Tensor(test_labels)

train_labels = dl.Tensor(np.eye(10)[train_labels.data.reshape(-1)])

np.random.seed(20)

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1), # Nx28x28x8
    nn.MaxPool2d(kernel_size=2, stride=2), # Nx14x14x8
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1), # Nx14x14x16
    nn.MaxPool2d(kernel_size=2, stride=2), # Nx7x7x16
    nn.Flatten(), # Nx784
    nn.Linear(in_features=784, out_features=10) # Nx10
)

optimizer = dl.optim.Adam()
loss_fn = nn.CrossEntropyLoss()
batch_size = 1024

epoch_loss_history = dl.train(
    model=model, train_data=train_data,
    train_labels=train_labels,
    optimizer=optimizer, loss_fn=loss_fn,
    batch_size=batch_size, num_epochs=10
)

test_pred = model(test_data)
test_pred = np.array([np.argmax(pred) for pred in test_pred.data])

num_correct = sum(
    test_pred == test_labels.data.reshape(-1)
)

print(f"Test Accuracy: {100*num_correct/test_labels.shape[0]}%")
