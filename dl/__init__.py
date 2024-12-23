from typing import List
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from . import functions
from . import nn
from . import optim

from dl.tensor import Tensor, Dependency
from dl.variable import Variable

def train(
    model: nn.Module,
    train_data: Tensor,
    train_labels: Tensor,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    num_epochs: int,
    batch_size: int = 32,
    random_seed: int = 20
) -> List[float]:
    def train_batch(start: int) -> float:
        end = start + batch_size
        inputs, outputs = train_data[start:end], train_labels[start:end]

        model.zero_grad()
        predicted = model(inputs)

        loss = loss_fn(predicted, outputs)
        loss.backward()
        optimizer.step(model)

        return loss.data

    np.random.seed(random_seed)
    starts, epoch_loss_history = np.arange(0, train_data.shape[0], batch_size), []

    with tqdm(total=num_epochs, colour="cyan") as progress_bar:
        for epoch in range(num_epochs):
            progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

            np.random.shuffle(starts)
            epoch_loss = sum(
                train_batch(int(start)) for start in starts
            )

            epoch_loss_history.append(epoch_loss)
            progress_bar.set_postfix({"Loss": f"{epoch_loss: .4f}"})
            progress_bar.update(1)
    
    return epoch_loss_history

def split(
    tensor: Tensor,
    sections: int,
    axis: int
) -> List[Tensor]:
    data_arrays = list(np.split(tensor.data, sections, axis=axis))
    requires_grads = [tensor.requires_grad for _ in range(sections)]

    depends_ons: List[List[Dependency]] = [[] for _ in range(sections)]
    tensors: List[Tensor] = []

    for i in range(sections):
        if requires_grads[i]:
            def grad_fn(grad: NDArray, section=i) -> NDArray:
                merging_grads = [
                    np.zeros_like(data_arrays[j]) if j != section else grad
                    for j in range(sections)
                ]
                print(merging_grads)

                return np.concatenate(merging_grads, axis=axis)
            depends_ons[i].append(Dependency(tensor, grad_fn))

        tensors.append(Tensor(data_arrays[i], requires_grads[i], depends_ons[i]))

    return tensors

__all__ = [
    "Tensor",
    "Dependency",
    "Variable"
]
