import sys, os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

import numpy as np
import dl
import dl.nn as nn

train_data, train_labels = np.load("./data/train_images_flat.npy"), np.load("./data/train_labels.npy")
test_data, test_labels = np.load("./data/test_images_flat.npy"), np.load("./data/test_labels.npy")

train_data, train_labels = dl.Tensor(train_data), dl.Tensor(train_labels)
test_data, test_labels = dl.Tensor(test_data), dl.Tensor(test_labels)

train_labels = dl.Tensor(np.eye(10)[train_labels.data.reshape(-1)])

print(f"Train Data: {train_data.shape}, Train Labels: {train_labels.shape}")
print(f"Test Data {test_data.shape}, Test Labels: {test_labels.shape}")

model = nn.Sequential(
    nn.Linear(in_features=784, out_features=512),
    nn.ReLU(),
    nn.Linear(in_features=512, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=10),
    nn.Softmax()
)

optimizer = dl.optim.Adam()
loss_fn = nn.CrossEntropyLoss()
batch_size = 64

epoch_loss_history = dl.train(
    model=model, train_data=train_data,
    train_labels=train_labels,
    optimizer=optimizer, loss_fn=loss_fn,
    num_epochs=10
)

test_pred = model(test_data)
test_pred = np.array([np.argmax(pred) for pred in test_pred.data])

num_correct = sum(
    test_pred == test_labels.data.reshape(-1)
)

print(f"Test Accuracy: {100*num_correct/test_labels.shape[0]}%")
