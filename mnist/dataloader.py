import struct
import numpy as np

def read_images(filename):
    with open(filename, 'rb') as f:
        _, num_items, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_items, rows * cols)
        return data

def read_labels(filename):
    with open(filename, 'rb') as f:
        _, num_items = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

train_images_path = "data/MNIST/raw/train-images-idx3-ubyte"
train_labels_path = "data/MNIST/raw/train-labels-idx1-ubyte"
test_images_path = "data/MNIST/raw/t10k-images-idx3-ubyte"
test_labels_path = "data/MNIST/raw/t10k-labels-idx1-ubyte"

train_images = read_images(train_images_path)
train_labels = read_labels(train_labels_path)
train_labels = train_labels.reshape(len(train_labels), 1)

test_images = read_images(test_images_path)
test_labels = read_labels(test_labels_path)
test_labels = test_labels.reshape(len(test_labels), 1)

train_images_flat = train_images.reshape(-1, 784)
test_images_flat = test_images.reshape(-1, 784)

print(f"Train images: {train_images_flat.shape} Train labels: {train_labels.shape}")
print(f"Test images shape: {test_images_flat.shape} Train labels: {train_labels.shape}")

np.save("./data/train_images_flat.npy", train_images_flat)
np.save("./data/train_labels.npy", train_labels)
np.save("./data/test_images_flat.npy", test_images_flat)
np.save("./data/test_labels.npy", test_labels)

print("Data saved successfully as numpy arrays!")
