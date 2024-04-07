r"""°°°
# Imports #
°°°"""
#|%%--%%| <4VecPzTeXr|9isZC9wadi>

import pennylane as qml
from pennylane import numpy as np
# import matplotlib as mlp
import networkx as nx

#|%%--%%| <9isZC9wadi|91fX6kWCua>
r"""°°°
# Methods #
°°°"""
#|%%--%%| <91fX6kWCua|gcNQ04GyOI>


# print('------------------------------------------------------------')
# print(f"Optimal expected value = {opt_obj}")
# print(f"Optimal parameters = {opt_params}")
# print(f"Optimal graph cut = {opt_cut}")

def find_ground_state(n_layers, graph=None, hamiltonian=None):
    n_qubits = 8

    dev_c = qml.device('default.qubit', wires=n_qubits, shots=1000)
    dev_e = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev_e)
    def qaoa_expval(params, n_layers, graph=None, hamiltonian=None):
        if graph is not None:
            C = qml.qaoa.maxcut(graph)[0]
        elif hamiltonian is not None:
            if isinstance(hamiltonian, (np.ndarray, qml.numpy.tensor)):
                C = qml.pauli_decompose(hamiltonian)
            else:
                C = hamiltonian
        qaoa_ec(params, n_layers, graph, hamiltonian)
        return qml.expval(C)

    @qml.qnode(dev_c)
    def qaoa_counts(params, n_layers, graph=None, hamiltonian=None):
        qaoa_ec(params, n_layers, graph, hamiltonian)
        return qml.counts()

    @qml.qnode(dev_e)
    def qaoa_state(params, n_layers, graph=None, hamiltonian=None):
        qaoa_ec(params, n_layers, graph, hamiltonian)
        return qml.state()

    @qml.qnode(dev_e)
    def get_state(gs):
        qml.BasisState(gs, wires=range(n_qubits))
        return qml.state()

    def qaoa_ec(params, n_layers, graph=None, hamiltonian=None):
        gammas = params[:n_layers]
        betas = params[n_layers:]

        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        if graph is not None:
            C, B = qml.qaoa.maxcut(graph)

            for p in range(n_layers):
                for (u, v) in list(graph.edges):
                    qml.IsingZZ(gammas[p] * graph[u][v]['weight'], wires=[u, v])

                qml.qaoa.mixer_layer(betas[p], B)

        elif hamiltonian is not None:
            if isinstance(hamiltonian, (np.ndarray, qml.numpy.tensor)):
                C = qml.pauli_decompose(hamiltonian)
            else:
                C = hamiltonian

            for p in range(n_layers):
                qml.qaoa.cost_layer(gammas[p], C)

                qml.qaoa.mixer_layer(betas[p], qml.qaoa.x_mixer(wires=range(n_qubits)))

    def optimize_angles(n_layers, graph=None, hamiltonian=None):
        init_params = np.random.uniform(low=-np.pi, high=np.pi, size=2*n_layers, requires_grad=True)
        params = init_params

        conv_tol = 1e-04
        max_steps = 100
        opt = qml.AdagradOptimizer(stepsize=0.4)

        paramss = [params]
        costs = [qaoa_expval(params, n_layers, graph, hamiltonian)]

        for step in range(max_steps):
            current_step = opt.step_and_cost(qaoa_expval, params, n_layers, graph, hamiltonian)
            params = current_step[0][0]
            prev_cost = current_step[1]
            paramss.append(params)
            costs.append(qaoa_expval(params, n_layers, graph, hamiltonian))

            # print("Cost after step {:5d}: {: .7f}".format(step + 1, -qaoa_expval(params, n_layers, graph, hamiltonian)))

            conv = np.abs(costs[-1] - prev_cost)

            if conv <= conv_tol:
                break

        # print("\nOptimized rotation angles: {}".format(paramss[-1]))
        return -costs[-1], paramss[-1]

    opt_obj, opt_params = optimize_angles(n_layers, graph, hamiltonian)

    counts = qaoa_counts(opt_params, n_layers, graph, hamiltonian)
    opt_cut = max(counts, key=counts.get)

    ground_state = qaoa_state(opt_params, n_layers, graph, hamiltonian)
    bit = get_state(np.array([eval(i) for i in opt_cut], requires_grad=False))

    return ground_state, bit

# gs = find_ground_state(1, graph=nx.cycle_graph(8))
# print(gs)
