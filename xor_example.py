import dl
import dl.nn as nn
import numpy as np

np.random.seed(20)

train_data = np.random.uniform(low=-1, high=1, size=(200, 2))
train_labels = ((train_data[:, 0] * train_data[:, 1]) < 0).astype(int).reshape(200, 1)

test_data = np.random.uniform(low=-1, high=1, size=(50, 2))
test_labels = ((test_data[:, 0] * test_data[:, 1]) < 0).astype(int).reshape(50, 1)

train_data, train_labels = dl.Tensor(train_data), dl.Tensor(train_labels)
test_data, test_labels = dl.Tensor(test_data), dl.Tensor(test_labels)

model = nn.Sequential(
    nn.Linear(in_features=2, out_features=4),
    nn.ReLU(),
    nn.Linear(in_features=4, out_features=4),
    nn.ReLU(),
    nn.Linear(in_features=4, out_features=1),
    nn.Sigmoid()
)

optimizer = dl.optim.Adam()
loss_fn = nn.BCELoss()
batch_size = 32

starts = np.arange(0, train_data.shape[0], batch_size)
for epoch in range(1000):
    epoch_loss = 0.0

    np.random.shuffle(starts)
    for start in starts:
        end = start + batch_size
        inputs, outputs = train_data[start: end], train_labels[start: end]

        model.zero_grad()
        predicted = model(inputs)

        loss = loss_fn(predicted, outputs)
        loss.backward()
        epoch_loss += loss.data

        optimizer.step(model)

    if (epoch+1)%100 == 0 or epoch == 0: print(f"Epoch[{epoch+1:04}/1000]\tLoss: {epoch_loss}")

test_pred = model(test_data)
print(f"\nTest Accuracy: {sum(
    np.where(test_pred.data.reshape(-1) > 0.5, 1, 0) ==
    test_labels.data.reshape(-1)
)*2}%")