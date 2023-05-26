import math
from math import pi
import pickle
import os
import statistics as stats

import torch
from torch import nn
from torch.nn import AvgPool2d, Flatten, Linear, ReLU, CrossEntropyLoss, Softmax
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchquantum as tq
import torchquantum.functional as tqf
from torchinfo import summary

from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


DATAFILE = "../deepsat_qnn/deepsat4/sat-4-full.mat"  # https://csc.lsu.edu/~saikat/deepsat/
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 40

DOWNSAMPLED_SIZE = 3
N_QUANTUM_KERNELS = 2

seed = 0
torch.manual_seed(seed)


class Data(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.Tensor(x_data).permute((3, 2, 0, 1))

        # Take the channel-wise mean of the data to reduce the dimensionality
        self.x_data = self.x_data.mean(dim=1, keepdim=True)

        # # Standardize the data on [0, pi]
        mn, mx = self.x_data.min(), self.x_data.max()
        self.x_data = pi * (self.x_data - mn) / (mx - mn)

        self.y_data = torch.Tensor(y_data).permute((1, 0))

        self.len = len(self.x_data)

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i]


class QuantumConvolution(tq.QuantumModule):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits

        # Instantiate the gates with trainable parameters for each kernel
        self.kernel_gates = [
            tq.RY(has_params=True, trainable=True),
            tq.RY(has_params=True, trainable=True),
            tq.RY(has_params=True, trainable=True),
            tq.RX(has_params=True, trainable=True),
            tq.RX(has_params=True, trainable=True),
        ]

    def forward(self, qdev):
        # Apply the quantum kernel
        ry0, ry1, ry2, rx0, rx1 = self.kernel_gates

        # Apply each quantum convolution block
        for i in range(self.n_qubits - 4):
            ry0(qdev, wires=i)
            qdev.cnot(wires=[i, i + 3])
            ry1(qdev, wires=i + 3)
            ry2(qdev, wires=i + 1)
            qdev.cnot(wires=[i + 1, i + 4])
            qdev.cnot(wires=[i + 3, i + 1])
            rx0(qdev, wires=i + 4)
            rx1(qdev, wires=i + 1)

        # Apply the pooling operations
        for i in range(self.n_qubits - 4):
            qdev.cnot(wires=[i + 1, i])
            qdev.cnot(wires=[i + 3, i])
            qdev.cnot(wires=[i + 4, i])


class QuantumModel(tq.QuantumModule):
    def __init__(self, batch_size, n_qubits, n_kernels):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_kernels = n_kernels
        self.batch_size = batch_size

        # Create a quantum device to run the gates
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits, bsz=self.batch_size)

        # Define the encoder to encode the input values using an Ry rotation
        # Input values should be scaled to [0, pi]
        self.encode = tq.GeneralEncoder([{'input_idx': [i], 'func': 'ry', 'wires': [i]} for i in range(self.n_qubits)])

        # Define the quantum kernels
        self.quantum_kernels = [QuantumConvolution(self.n_qubits) for _ in range(self.n_kernels)]

        # Register the kernel's parameters
        for i, kernel in enumerate(self.quantum_kernels):
            for j, gate in enumerate(kernel.kernel_gates):
                self.register_parameter(name=f'kernel{i}_gate{j}', param=gate.params)

        # Measure all gates to obtain expectation values of all qubits
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        # Make sure the input has the right shape
        # assert x.shape[0] == self.batch_size, 'Incorrect input batch size'
        assert x.shape[1] == self.n_qubits, 'Input shape does not match number of qubits in the circuit'

        # Apply the quantum kernels
        quantum_conv_results = []
        for kernel_block in self.quantum_kernels:
            # Encode the data
            self.encode(self.q_device, x)

            # Apply the quantum kernel block
            kernel_block(self.q_device)

            # Obtain the expectation values of the qubits in the computational basis after quantum convolution
            # Here we disregard the last four qubits due to the pooling operation
            quantum_conv_results.append(self.measure(self.q_device)[:, :-4])

            # Reset the states of the quantum device to |0>
            self.q_device.reset_states(bsz=self.batch_size)

        return torch.cat(quantum_conv_results, dim=1)


class QuantumDropout(tq.QuantumModule):
    def __init__(self, p: float, input_shape: tuple):
        super().__init__()

        self.p = p  # Proportion of the input vector to set to zero
        self.batch_size = input_shape[0]
        self.input_length = math.prod(input_shape[1:])

        # Number of wires in the circuit is the minimal m >= 1 s.t. 2**m >= self.input_length
        self.n_wires = math.ceil(math.log2(self.input_length)) or 1

    def forward(self, x):
        # Define a quantum device to run the encoding
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=self.batch_size)

        # Flatten x if necessary
        shape = x.shape
        if len(shape) > 2:
            x = x.flatten(start_dim=1)

        # Pass the input tensor through the quantum circuit defined by the encoder gates
        # Quantum Circuit:
        # -- H -- Ry(x_1) -- H -- Ry(x_m+1) -- H -- ...
        # -- H -- Ry(x_2) -- H -- Ry(x_m+2) -- H -- ...
        # ...
        # -- H -- Ry(x_m) -- H -- Ry(x_2*m) -- H -- ...
        # where m is minimal s.t. 2**m >= # of inputs.

        # First initialize each qubit in the |+> state
        for i in range(self.n_wires):
            tqf.h(qdev, wires=i)

        # Then apply the rotation gates and remaining hadamards
        for i in range(self.input_length):
            tqf.ry(qdev, wires=i % self.n_wires, params=x[:, i])
            tqf.h(qdev, wires=i % self.n_wires)

        # Get the elements of the statevector corresponding to the input tensor
        statevector = qdev.get_states_1d()[:, : self.input_length]

        # We just want the magnitude
        statevector = torch.real(statevector * statevector.conj())

        # Number of elements to set to zero in each input of the batch
        n_zero = round(self.p * self.input_length)

        # Get indices of the largest elements of the statevector
        topk = torch.topk(statevector, n_zero, dim=1).indices

        # Create a mask and zero the corresponding input values
        mask = torch.ones_like(x)
        for i in range(self.batch_size):
            mask[i, topk[i]] = 0.0
        x = x * mask  # Cannot use *= as this operates in-place, which causes errors for torch.autograg

        # Reshape x if necessary
        if len(shape) > 2:
            x = x.reshape(shape)

        return x


class HybridModel(nn.Module):
    def __init__(self, input_size=28, downsampled_size=4, quantum_kernels=2):
        super().__init__()

        # Downsample the image from [BS, 1, 28, 28] to [BS, 1, downsampled_size, downsampled_size] and flatten it
        downsampling_ks = input_size // downsampled_size
        self.downsampling = AvgPool2d(kernel_size=downsampling_ks, stride=downsampling_ks)
        self.flatten = Flatten()

        # Apply the quantum layer to the flattened tensor
        self.quantum_convolution = QuantumModel(
            batch_size=BATCH_SIZE, n_qubits=downsampled_size**2, n_kernels=quantum_kernels
        )

        # Feed the quantum output into linear layers
        self.fc1 = Linear(in_features=quantum_kernels * (downsampled_size**2 - 4), out_features=128)
        self.quantum_dropout = QuantumDropout(0.4, (BATCH_SIZE, 128))
        self.fc2 = Linear(in_features=128, out_features=64)
        self.fc3 = Linear(in_features=64, out_features=4)

        # Activation function for linear layers
        self.relu = ReLU()

    def forward(self, x):
        x = self.downsampling(x)
        x = self.flatten(x)
        x = self.quantum_convolution(x)
        x = self.relu(self.fc1(x))
        # x = self.quantum_dropout(x)
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


def main(plot=True):
    train_loader, test_loader = load_data(subset_directory='data_subsets')

    # plot_samples(train_loader, DOWNSAMPLED_SIZE)
    hybrid_model = HybridModel(input_size=28, downsampled_size=DOWNSAMPLED_SIZE, quantum_kernels=N_QUANTUM_KERNELS)

    # Summarize the model
    summary(hybrid_model, (BATCH_SIZE, 1, 28, 28))

    # Define the optimizer and loss function
    optimizer = Adam(hybrid_model.parameters(), lr=LR)
    loss_func = CrossEntropyLoss()

    try:
        train_loss, test_loss = [], []
        train_acc, test_acc = [], []
        for i in tqdm(range(EPOCHS)):
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
    if plot:
        plt.figure()
        sns.lineplot(train_loss, label='train')
        sns.lineplot(test_loss, label='test')
        plt.title('Loss')

        plt.figure()
        sns.lineplot(train_acc, label='train')
        sns.lineplot(test_acc, label='test')
        plt.title('Accuracy')

        plt.show()

    # print(train_loss, train_acc, test_loss, test_acc, sep='\n')
    return train_loss, train_acc, test_loss, test_acc


def run_many(n=4):
    mean_results = torch.zeros((4, EPOCHS), requires_grad=False)
    for i in range(n):
        print(f'Beginning run {i+1}/{n}')
        mean_results += torch.Tensor(main(plot=False))
    mean_results /= n

    print('Mean results:')
    for result in mean_results:
        print(list(result))

    train_loss, train_acc, test_loss, test_acc = mean_results
    plt.figure()
    sns.lineplot(train_loss, label='train')
    sns.lineplot(test_loss, label='test')
    plt.title('Loss')

    plt.figure()
    sns.lineplot(train_acc, label='train')
    sns.lineplot(test_acc, label='test')
    plt.title('Accuracy')

    plt.show()


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
    run_many(4)
