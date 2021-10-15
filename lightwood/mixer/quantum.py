from typing import Dict
import pandas as pd
import numpy as np
from torch import Tensor

from lightwood.data.encoded_ds import EncodedDs
from lightwood.api.types import PredictionArguments
from lightwood.encoder.base import BaseEncoder
from lightwood.mixer import BaseMixer
from lightwood import dtype

################################################################################

import torch
from torch.autograd import Function
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import qiskit
from qiskit import transpile, assemble


class QuantumCircuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """

    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)

        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        t_qc = transpile(self._circuit,
                         self.backend)
        qobj = assemble(t_qc,
                        shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)

        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)

        return np.array([expectation])


class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])

            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None


class QuantumNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(QuantumNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.hybrid = Hybrid(qiskit.Aer.get_backend('aer_simulator'), 100, np.pi / 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x: Tensor = self.fc2(x)

        assert x.size() == torch.Size((1, 5)), x.size()

        # TODO(Andrea): there must be a better way to do this
        x = torch.stack([self.hybrid(t.reshape((1, 1))) for t in x.view(-1)]).reshape((1, 5))

        x = self.softmax(x)

        assert x.size() == torch.Size((1, 5)), x.size()

        return x


class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """

    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


################################################################################


class QuantumMixer(BaseMixer):
    stop_after: int
    supports_proba: bool = False
    target_encoder: BaseEncoder
    stable: bool = True  # TODO(Andrea): what does this even mean

    def __init__(
        self, stop_after: int, target: str, dtype_dict: Dict[str, str],
        target_encoder: BaseEncoder
    ):
        super().__init__(stop_after)

        # TODO(Andrea): parse dtype_dict?
        if dtype_dict[target] not in (dtype.categorical, dtype.binary):
            raise Exception('This mixer can only be used for classification problems!'
                            f'Got target dtype {dtype_dict[target]} instead!')

        self.stop_after = stop_after
        self.supports_proba = False
        self.target_encoder = target_encoder

    def fit(self, train_data: EncodedDs, _dev_data: EncodedDs) -> None:
        # TODO(Andrea): QuantumNet requires the batch size to be 1
        train_dl = DataLoader(train_data, batch_size=1, shuffle=False)

        # TODO(Andrea): find a reasonable number
        epochs = 5
        loss_list = []

        self.model = QuantumNet(
            input_dim=len(train_data[0][0]),
            output_dim=len(train_data[0][1])
        )

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_func = nn.NLLLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = []
            for _, (data, target) in enumerate(train_dl):
                assert data.size() == torch.Size((1, 24))
                assert target.size() == torch.Size((1, 5))

                optimizer.zero_grad()
                # Forward pass
                output: Tensor = self.model(data)
                # Calculating loss
                loss = loss_func(output, target.argmax(1))
                # Backward pass
                loss.backward()
                # Optimize the weights
                optimizer.step()

                total_loss.append(loss.item())
            loss_list.append(sum(total_loss) / len(total_loss))
            print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
                100. * (epoch + 1) / epochs, loss_list[-1]))

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        # Iterate through the dataset and add the batch_size dimension
        X = [x.unsqueeze(0) for x, _ in ds]

        # Evaluate the model for all elements of the datasets, separately (i.e.)
        # batch_size=1
        self.model = self.model.eval()
        with torch.no_grad():
            Yh = [self.model(x) for x in X]

        # The model returns per-class log-probabilities
        # Convert them to indices
        Yh = Tensor([yh.argmax() for yh in Yh]).to(torch.int64)

        # Convert the indices to OHE
        Yh = F.one_hot(Yh, num_classes=5)
        decoded_predictions = self.target_encoder.decode(Yh)

        return pd.DataFrame({'prediction': decoded_predictions})

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        pass
