import torch
import qiskit
import numpy as np

from lightwood.mixer.helpers.default_net import DefaultNet
from lightwood.helpers.torch import LightwoodAutocast


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
        job = self.backend.run([self._circuit.bind_parameters({self.theta: t})
                                for t in thetas], shots=self.shots)
        results = job.result().get_counts()

        final = []
        for result in results:
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)

            # Compute probabilities for each state
            probabilities = counts / self.shots

            # Get state expectation
            expectation = np.sum(states * probabilities)
            final.append(expectation)
        return np.array(final)


class HybridSingleFunction(torch.autograd.Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input.tolist())
        result = torch.tensor(expectation_z)
        ctx.save_for_backward(input, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        expectation_left = ctx.quantum_circuit.run(shift_left)
        expectation_right = ctx.quantum_circuit.run(shift_right)
        gradients = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
        gradients = np.array(gradients).T
        return torch.tensor(gradients).float() * grad_output.float(), None, None


class HybridSingle(torch.nn.Module):
    """ Hybrid quantum - classical layer definition """

    def __init__(self, backend, shots, shift):
        super(HybridSingle, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridSingleFunction.apply(input, self.quantum_circuit, self.shift)


class QClassicNet(DefaultNet):
    """
    DefaultNet variant that uses qiskit to add a final quantum layer
    """

    def __init__(self,
                 input_size: int = None,
                 output_size: int = None,
                 shape: list = None,
                 max_params: int = 3e7,
                 num_hidden: int = 1,
                 dropout: float = 0) -> None:
        hidden_size = max([input_size * 2, output_size * 2, 400])
        super().__init__(input_size=input_size,
                         output_size=hidden_size,
                         shape=shape,
                         max_params=max_params,
                         num_hidden=num_hidden,
                         dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.hybrid = HybridSingle(qiskit.Aer.get_backend('aer_simulator'), 100, np.pi / 2)

    def to(self, device=None, available_devices=None):
        return super().to(device)

    def forward(self, input):
        with LightwoodAutocast():
            if len(input.shape) == 1:
                input = input.unsqueeze(0)
            classical_output = self.fc(self.net(input))
            full_output = torch.stack([self.hybrid(i) for i in classical_output])
        return full_output.float()
