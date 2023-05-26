from math import pi
import pickle
import os
import statistics as stats

import torch
from torch import nn
from torch.nn import AvgPool2d, Flatten, Linear, ReLU, CrossEntropyLoss, Softmax
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# import torchquantum as tq
from torchinfo import summary

from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.opflow import PauliSumOp


DATAFILE = "../deepsat_qnn/deepsat4/sat-4-full.mat"  # https://csc.lsu.edu/~saikat/deepsat/
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 100

DOWNSAMPLED_SIZE = 3
N_QUANTUM_KERNELS = 2

# seed = 0
# torch.manual_seed(seed)


class Data(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.Tensor(x_data).permute((3, 2, 0, 1))

        # Take the channel-wise mean of the data to reduce the dimensionality
        self.x_data = self.x_data.mean(dim=1, keepdim=True)

        # Standardize the data on [0, pi]
        mn, mx = self.x_data.min(), self.x_data.max()
        self.x_data = pi * (self.x_data - mn) / (mx - mn)

        self.y_data = torch.Tensor(y_data).permute((1, 0))

        self.len = len(self.x_data)

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i]


def create_qnn_kernel(n_qubits):
    # Create the input feature encoding circuit
    feature_map = QuantumCircuit(n_qubits)
    input_parameters = ParameterVector('inputs', n_qubits)
    for i in range(n_qubits):
        feature_map.ry(input_parameters[i], i)

    # Create the convolution kernel circuit
    conv_circuit = QuantumCircuit(n_qubits)
    conv_parameters = ParameterVector('weights', 5)
    for i in range(n_qubits - 4):
        conv_circuit.ry(conv_parameters[0], i)
        conv_circuit.cx(i, i + 3)
        conv_circuit.ry(conv_parameters[1], i + 3)
        conv_circuit.ry(conv_parameters[2], i + 1)
        conv_circuit.cx(i + 1, i + 4)
        conv_circuit.cx(i + 3, i + 1)
        conv_circuit.rx(conv_parameters[3], i + 4)
        conv_circuit.rx(conv_parameters[4], i + 1)
        conv_circuit.barrier()

    # Apply the pooling operation
    for i in range(n_qubits - 4):
        conv_circuit.cx(i + 1, i)
        conv_circuit.cx(i + 3, i)
        conv_circuit.cx(i + 4, i)
        conv_circuit.barrier()

    return EstimatorQNN(
        circuit=feature_map.compose(conv_circuit),
        input_params=feature_map.parameters,
        weight_params=conv_circuit.parameters,
        observables=[
            PauliSumOp.from_list([(''.join(['Z' if i == j else 'I' for i in range(n_qubits)]), 1)])
            for j in range(4, n_qubits)
        ],  # Second range starts at 4 as the pooling operation disregards the final 4 qubits (deals with qiskit's stupid qubit ordering)
        input_gradients=True,
    )


class HybridModel(nn.Module):
    def __init__(self, input_size=28, downsampled_size=4, n_quantum_kernels=2):
        super().__init__()

        # Downsample the image from [BS, 1, 28, 28] to [BS, 1, downsampled_size, downsampled_size] and flatten it
        downsampling_ks = input_size // downsampled_size
        self.downsampling = AvgPool2d(kernel_size=downsampling_ks, stride=downsampling_ks)
        self.flatten = Flatten()

        # Apply the quantum layer to the flattened tensor
        self.quantum_kernels = [
            TorchConnector(create_qnn_kernel(downsampled_size**2)) for _ in range(n_quantum_kernels)
        ]

        # Register the parameters of the quantum layer
        for i, kernel in enumerate(self.quantum_kernels):
            for j, params in enumerate(kernel.parameters()):  # Should only be one
                self.register_parameter(f'kernel{i}_param{j}', params)

        # Feed the quantum output into linear layers
        self.fc1 = Linear(in_features=n_quantum_kernels * (downsampled_size**2 - 4), out_features=128)
        self.fc2 = Linear(in_features=128, out_features=64)
        self.fc3 = Linear(in_features=64, out_features=4)

        # Activation function for linear layers
        self.relu = ReLU()

    def apply_quantum_kernels(self, x):
        results = [kernel(x) for kernel in self.quantum_kernels]
        return torch.cat(results, dim=1)

    def forward(self, x):
        x = self.downsampling(x)
        x = self.flatten(x)
        x = self.apply_quantum_kernels(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # No need to apply softmax here as it is applied by the loss function


def load_data(ntrain=9000, ntest=1000, subset_directory='', write_subset_files=True):
    subset_files = [f'xtrain{ntrain}.pkl', f'xtest{ntest}.pkl', f'ytrain{ntrain}.pkl', f'ytest{ntest}.pkl']
    subset_files = [os.path.join(subset_directory, file) for file in subset_files]

    if all(os.path.exists(file) for file in subset_files):
        # If the specified data subsets have already been written to files, just read the files
        data = []
        for file in subset_files:
            with open(file, 'rb') as f:
                data.append(pickle.load(f))
        x_train, x_test, y_train, y_test = data
    else:
        # If the specified data subsets do not yet exist, read the data from the master file and write the subset files
        data = loadmat(DATAFILE)
        x_train, x_test, y_train, y_test = (
            data["train_x"][:, :, :, :ntrain],
            data["test_x"][:, :, :, :ntest],
            data["train_y"][:, :ntrain],
            data["test_y"][:, :ntest],
        )
        if write_subset_files:
            for file, data in zip(subset_files, [x_train, x_test, y_train, y_test]):
                with open(file, 'wb') as f:
                    pickle.dump(data, f)

    # Preprocess the data
    train_data = Data(x_train, y_train)
    test_data = Data(x_test, y_test)

    # Define the dataloaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train(model, dataloader, loss_func, optimizer, epoch=0):
    train_loss, train_accuracy = [], []
    softmax = Softmax(dim=1)

    model.train()
    for x, y in tqdm(dataloader, desc=f'Training epoch {epoch + 1}/{EPOCHS}'):
        if x.shape[0] != BATCH_SIZE:
            continue
        # Zero gradients and compute the prediction
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)

        # Loss computation and backpropagation
        loss = loss_func(prediction, y)
        loss.backward()

        # Parameter optimization
        optimizer.step()

        # Track loss and accuracy metrics
        train_loss.append(loss.item())
        train_accuracy.append(
            (torch.argmax(y, dim=1) == torch.argmax(softmax(prediction), dim=1)).sum().item() / len(y)
        )

    return train_loss, train_accuracy


@torch.no_grad()
def test(model, dataloader, loss_func, epoch=0):
    test_loss, test_accuracy = [], []
    softmax = Softmax(dim=1)

    model.eval()
    for x, y in tqdm(dataloader, desc=f'Training epoch {epoch + 1}/{EPOCHS}'):
        if x.shape[0] != BATCH_SIZE:
            continue
        # Obtain predictions and track loss and accuracy metrics
        prediction = model(x)
        test_loss.append(loss_func(prediction, y).item())
        test_accuracy.append(
            (torch.argmax(y, dim=1) == torch.argmax(softmax(prediction), dim=1)).sum().item() / len(y)
        )

    return test_loss, test_accuracy


def main():
    train_loader, test_loader = load_data(subset_directory='data_subsets')

    # plot_samples(train_loader, DOWNSAMPLED_SIZE)
    hybrid_model = HybridModel(input_size=28, downsampled_size=DOWNSAMPLED_SIZE, n_quantum_kernels=N_QUANTUM_KERNELS)

    # Summarize the model
    summary(hybrid_model, (BATCH_SIZE, 1, 28, 28))

    # Define the optimizer and loss function
    optimizer = Adam(hybrid_model.parameters(), lr=LR)
    loss_func = CrossEntropyLoss()

    try:
        train_loss, test_loss = [], []
        train_acc, test_acc = [], []
        for i in range(EPOCHS):
            loss, acc = train(hybrid_model, train_loader, loss_func, optimizer, i)
            train_loss.append(stats.mean(loss))
            train_acc.append(stats.mean(acc))

            loss, acc = test(hybrid_model, test_loader, loss_func, i)
            test_loss.append(stats.mean(loss))
            test_acc.append(stats.mean(acc))
            print(
                f'Epoch {i + 1}/{EPOCHS}  |  '
                f'train loss {train_loss[-1]:.4f}  |  '
                f'train acc {train_acc[-1]:.2%}  |  '
                f'test loss {test_loss[-1]:.4f}  |  '
                f'test acc {test_acc[-1]:.2%}'
            )
    except KeyboardInterrupt as e:
        if not test_acc:
            raise KeyboardInterrupt(e)

    # Plot the results
    plt.figure()
    sns.lineplot(train_loss, label='train')
    sns.lineplot(test_loss, label='test')
    plt.title('Loss')

    plt.figure()
    sns.lineplot(train_acc, label='train')
    sns.lineplot(test_acc, label='test')
    plt.title('Accuracy')

    plt.show()

    print(train_loss, train_acc, test_loss, test_acc, sep='\n')


def plot_samples(dataloader, downsampled_size, n_samples=16, n_wide=4):
    import random

    n_samples += (n_wide - n_samples % n_wide) % n_wide
    samples = random.sample(list(dataloader), n_samples)

    downsampling_ks = samples[0][0].shape[-1] // downsampled_size
    downsampling = AvgPool2d(kernel_size=downsampling_ks, stride=downsampling_ks)

    plt.figure(figsize=(12, 10))
    with torch.no_grad():
        for i in range(n_samples):
            plt.subplot(n_samples // n_wide, n_wide, i + 1)
            sample = downsampling(samples[i][0])
            class_label = str(torch.argmax(samples[i][1]).item())
            plt.imshow(sample[0, 0], cmap='gray')
            plt.title(f'Class: {class_label}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
