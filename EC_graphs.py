# TODO: test how many iterations of QAOA are needed to get the same result as EC
# TODO: see which subgraphs are best for EC
# TODO: use triangle-free subgraphs
# TODO: plot energy landscape

import numpy as np
import math
import cmath as cm
import scipy
from scipy import linalg
from scipy.linalg import expm
from ec_qaoa import find_ground_state
import pennylane as qml
import networkx as nx
import pandas as pd
# from rich_dataframe import prettify
import os
import matplotlib.pyplot as plt

from functools import partial
from os import path
from multiprocessing import Pool
from tqdm import tqdm
from natsort import natsort_keygen

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

#|%%--%%| <NRy8NDo3j4|1O7VJ9sp9z>

paulis = {}
paulis['X'] = np.array([[0, 1], [1, 0]], dtype=complex)
paulis['Y'] = np.array([[0, -1.j], [1.j, 0]], dtype=complex)
paulis['Z'] = np.array([[1, 0], [0, -1]], dtype=complex)
paulis['I'] = np.array([[1, 0], [0, 1]], dtype=complex)


def many_kron(ops):
    # Takes an array of Pauli characters and produces the tensor product
    op = paulis[ops[0]]
    if len(ops) == 1:
        return op

    for opj in ops[1:]:
        op = np.kron(op, paulis[opj])

    return op

# |%%--%%| <1O7VJ9sp9z|ARj5emkipt>


def ising_ham(N):
    Jmat = Jmatrix(N)
    ham = np.zeros([2**N, 2**N], dtype=complex)
    hamlist = []

    # Build hamiltonian matrix
    for isite in range(N):
        oplist = ['I']*N
        oplist[isite] = 'Z'
        for jsite in range(N-1-isite):
            oplist[isite] = 'Z'
            ham += Jmat[isite, jsite]*many_kron(oplist)
    return ham


def ising_ham_pauli_string(N):
    Jmat = Jmatrix(N)
    ham = qml.Identity(wires=range(N))

    # Build hamiltonian matrix
    for isite in range(N):
        oplist = ['I']*N
        oplist[isite] = 'Z'
        for jsite in range(N-1-isite):
            oplist[isite] = 'Z'
            ham += Jmat[isite, jsite]*qml.pauli.string_to_pauli_word(''.join(oplist))
    return ham


def Jmatrix(N):
    # Jmat = np.zeros([N-1, N-1], dtype=float)
    Jmat = np.random.randint(2, size=(N-1, N-1))
    return Jmat

# |%%--%%| <ARj5emkipt|0rfwybvP8h>


def do_continuation(ham, basis):

    nvecs = len(basis)

    smaller_ham = np.zeros([nvecs, nvecs], dtype=complex)
    overlap_matrix = np.zeros_like(smaller_ham)

    for i in range(nvecs):
        ui = basis[i, :]
        # ui = basis[i]

        for j in range(i, nvecs):
            uj = basis[j, :]
            # uj = basis[j]

            smaller_ham[i, j] = np.conjugate(np.transpose(ui)) @ ham @ uj

            if not i == j:
                smaller_ham[j, i] = np.conjugate(smaller_ham[i, j])

            overlap_matrix[i, j] = np.dot(np.conjugate(np.transpose(ui)), uj)

            if not i == j:
                overlap_matrix[j, i] = np.conjugate(overlap_matrix[i, j])
    evals, evecs = scipy.linalg.eigh(smaller_ham, overlap_matrix)
    # print('Eigenvalues of S matrix', np.linalg.eigvalsh(smaller_ham))
    return evals, evecs


def get_new_evals(ham, basis):
    #print('condition number of the basis is=', np.linalg.cond(basis))
    nvecs = len(basis)
    evals, evecs = do_continuation(ham, basis)
    new_evals = np.zeros([nvecs], dtype='complex')
    cal_vec = np.zeros_like(basis[0])
    for k in range(nvecs):
        fullvec = np.zeros_like(basis[0])

        for l in range(nvecs):
            fullvec += evecs[l, k] * basis[l]
            # print('vector:', fullvec)
        energy = np.conjugate(np.transpose(fullvec)) @ ham @ fullvec
        if k == 0:
            cal_vec = fullvec   # ground state vector is stored for fidelity calculations
        new_evals[k] = energy
    #print('new evals', new_evals)
    return new_evals, cal_vec

#|%%--%%| <0rfwybvP8h|1SQEtBO6N9>

# bitstring to maxcut expected Value


def bitsting_to_maxcut(bitstring, graph):
    val = 0
    for (u, v) in graph.edges:
        if bitstring[u] != bitstring[v]:
            val += 1
    return val

#|%%--%%| <1SQEtBO6N9|W1ULrk2Lq2>


# N = 8
# num_basis = 5
# hams = [ising_ham_pauli_string(N) for _ in range(num_basis)]
# ec_ham = ising_ham(N)
# # print(hams)
#
# ground_states = np.array([find_ground_state(n_layers=1, hamiltonian=ham) for ham in hams])
# print(ground_states)
# print()
#
# evals, evec = get_new_evals(ec_ham, ground_states)
# # print(evals)
# print()
# # print(evec)
# print(f'{np.where(np.real(evec) == 1)[0][0]:0{N}b}')
#
#|%%--%%| <W1ULrk2Lq2|cnPwWYsExW>

# N = 8
# num_basis = 2
# num_graphs = 5
#
# # loop through 100 graphs, find ground state and expected value and store in array
#
#
# # NOTE: Passe in original graph, subgraph, and number of basis states
# def test_graphs(num_graphs, num_basis):
#     graph_index = np.random.randint(0, 11117, size=num_graphs)
#     graphs = [nx.read_gml(f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/graphs/main/all_8/graph_{graph_index[i]}/{graph_index[i]}.gml', destringizer=int) for i in range(num_graphs)]
#
#     results = pd.DataFrame(columns=['graph_index', 'case', 'bitstring', 'maxcut'])
#
#     for idx, graph in zip(graph_index, graphs):
#         i = 1
#         # NOTE: Change this to limit number of tries
#         while True:
#             try:
#                 # NOTE: Need this
#                 random_graphs = [graph.copy() for _ in range(num_basis)]
#                 subgraph = nx.read_gml(f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/graphs/main/all_8/graph_{idx}/pseudo_random/4.gml')
#                 for a, g in enumerate(random_graphs):
#                     # add weights to the edges that are in graph but not in g, and add those edges to g
#                     # NOTE: Need this
#                     for (u, v) in graph.edges:
#                         if (u, v) not in subgraph.edges:
#                             g.add_edge(u, v)
#                             g[u][v]['weight'] = a/(num_basis+1)
#                         else:
#                             g[u][v]['weight'] = 1
#
#                 # NOTE: Need this
#                 ec_hams = qml.matrix(qml.qaoa.maxcut(graph)[0])
#                 ground_states = np.array([find_ground_state(n_layers=1, graph=rg) for rg in random_graphs])
#
#                 evals, evec = get_new_evals(ec_hams, ground_states)
#
#             # NOTE: Need this
#             except linalg.LinAlgError:
#                 print(f'Try {i} failed for graph {idx}')
#                 i += 1
#                 continue
#             break
#
#         # NOTE: Want to compare with QAOA
#         ec_bit = f'{np.where(np.real(evec) == 1)[0][0]:0{N}b}'
#         qaoa_bit = f'{np.where(np.real(find_ground_state(1, graph=graph)) == 1)[0][0]:0{N}b}'
#         val, cut = nx.algorithms.approximation.maxcut.one_exchange(graph)
#         class_bit = ''.join(['1' if i in cut[0] else '0' for i in range(8)])
#
#         for case, bit in zip(['ec', 'qaoa', 'class'], [ec_bit, qaoa_bit, class_bit]):
#             results = pd.concat([results, pd.DataFrame([{
#                 'graph_index': str(idx),
#                 'case': case,
#                 'bitstring': bit,
#                 'maxcut': bitsting_to_maxcut(bit, graph)}])])
#
#     return results
#
#
# results = test_graphs(num_graphs, num_basis)

#|%%--%%| <cnPwWYsExW|vOIlPfkO3X>

# data_path = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/graphs/main/all_8/'
# out_path = '/home/vilcius/Papers/eigenvector_continuation/code/results.csv'


def init_dataframe(data_path: str, out_path: str):
    paths = [(f'{data_path}graph_{i}/{i}.gml', f'{data_path}graph_{i}/remove_triangle/all.gml')
             for i in range(11117) if path.exists(f'{data_path}graph_{i}/remove_triangle/all.gml')]
    index = pd.MultiIndex.from_tuples(paths, names=["path", "triangle_free_path"])
    df = pd.DataFrame(index=index)

    df.to_csv(out_path)


#|%%--%%| <vOIlPfkO3X|dniHaFRu2G>

# NOTE: Pass in original graph, subgraph, and number of basis states


def calc_EC(graph, subgraph, num_basis):

    i = 1
    # NOTE: Change this to limit number of tries
    while i < 5:
        try:
            # NOTE: Need this
            random_graphs = [subgraph.copy() for _ in range(num_basis)]
            new_random_graphs = []
            for a, g in enumerate(random_graphs):
                # add weights to the edges that are in graph but not in g, and add those edges to g
                # NOTE: Need this
                for (u, v) in graph.edges:
                    if (u, v) not in subgraph.edges:
                        g.add_edge(u, v)
                        g[u][v]['weight'] = a/(num_basis+1)
                    else:
                        g[u][v]['weight'] = 1
                new_random_graphs.append(g)

                # NOTE: Need this
            ec_hams = qml.matrix(qml.qaoa.maxcut(graph)[0])
            ground_states = np.array([find_ground_state(n_layers=1, graph=rg)[0] for rg in new_random_graphs])

            evals, evec = get_new_evals(ec_hams, ground_states)
            break

            # NOTE: Need this
        except linalg.LinAlgError:
            print(f'Try {i} failed for graph')
            i += 1

            if i == 5:
                evals = -1
                evec = -1
            continue
            break

    # NOTE: Want to compare with QAOA
    if i == 5:
        ec_bit = -1
        ec_maxcut = -1
    else:
        print("-------------This graph worked!------------")
        # ec_bit = f'{np.where(np.real(evec) == 1)[0][0]:0{8}b}'
        # ec_maxcut = bitsting_to_maxcut(ec_bit, graph)

    # qaoa_bit = f'{np.where(np.real(find_ground_state(1, graph=graph)[1]) == 1)[0][0]:0{8}b}'
    # val, cut = nx.algorithms.approximation.maxcut.one_exchange(graph)
    # class_bit = ''.join(['1' if i in cut[0] else '0' for i in range(8)])

    # qaoa_maxcut = bitsting_to_maxcut(qaoa_bit, graph)
    # class_maxcut = bitsting_to_maxcut(class_bit, graph)

    # return evals, evec, ec_bit, qaoa_bit, class_bit, ec_maxcut, qaoa_maxcut, class_maxcut
    qaoa_vec = find_ground_state(1, graph=graph)[0]
    return evals, evec, qaoa_vec  # ec_bit, class_bit, ec_maxcut, class_maxcut

#|%%--%%| <dniHaFRu2G|A6avno1hYp>


class Worker_EC():
    """
    Worker that executes random circuit QAOA for a given graph and graph_random.
    var random_type: The type of random graph to evaluate (random, pseudo_random, or remove_triangle).
    :var out_col: Name of the output column for AR.
    """

    def process_entry(self, entry: tuple[tuple[str, str], pd.Series]) -> pd.Series:
        paths, series = entry
        print(paths)
        graph = self.reader(paths[0])
        subgraph = self.reader(paths[1])

        # evals, evec, ec_bit, qaoa_bit, class_bit, ec_maxcut, qaoa_maxcut, class_maxcut = calc_EC(graph, subgraph, self.num_basis)
        evals, evec, qaoa_vec = calc_EC(graph, subgraph, self.num_basis)

        series['evals'] = evals
        series['evec'] = evec
        series['qaoa_vec'] = qaoa_vec
        # series['ec_bit'] = ec_bit
        # series['qaoa_bit'] = qaoa_bit
        # series['class_bit'] = class_bit
        # series['ec_maxcut'] = ec_maxcut
        # series['qaoa_maxcut'] = qaoa_maxcut
        # series['class_maxcut'] = class_maxcut
        return series


def optimize_ec_parallel(dataframe_path: str, rows_func: callable, num_workers: int, worker: Worker_EC):
    df = pd.read_csv(dataframe_path, index_col=[0, 1])
    selected_rows = rows_func(df)
    rows_to_process = list(df.loc[selected_rows, :].iterrows())
    # print(rows_to_process)
    remaining_rows = df.loc[~selected_rows, :]

    if len(rows_to_process) == 0:
        return

    results = []
    if num_workers == 1:
        for result in tqdm(map(worker.process_entry, rows_to_process), total=len(rows_to_process), smoothing=0, ascii=' █'):
            results.append(result)
    else:
        with Pool(num_workers) as pool:
            for result in tqdm(pool.imap(worker.process_entry, rows_to_process), total=len(rows_to_process), smoothing=0, ascii=' █'):
                results.append(result)

    df = pd.concat((pd.DataFrame(results), remaining_rows))  # .sort_index(key=natsort_keygen())
    df.index.names = ['path', 'triangle_free_path']
    df.to_csv(dataframe_path)


#|%%--%%| <A6avno1hYp|Lo8quqFE0x>

def run_ec_parallel():
    for num_basis in range(2, 6):
        p = 1
        convergence_threshold = 1e-4

        # NOTE: change these paths on server
        data_path = '/home/agwilkie/papers/random_circuit/MA-QAOA/graphs/main/all_8/'
        out_path = f'results_{num_basis}.csv'

        init_dataframe(data_path, out_path)

        num_workers = 20
        worker = Worker_EC()

        # reader = partial(nx.read_gml, destringizer=int)
        def rows_func(df): return np.ones((df.shape[0], 1), dtype=bool) if p == 1 else df[f'p_{p - 1}'] < convergence_threshold

        worker.num_basis = num_basis
        worker.reader = partial(nx.read_gml, destringizer=int)

        optimize_ec_parallel(out_path, rows_func, num_workers, worker)

#|%%--%%| <Lo8quqFE0x|OjeNgwivjv>


# prettify(results, row_limit=100)
#
# per_better = (results[results['case'] == 'ec']['maxcut'] > results[results['case'] == 'qaoa']['maxcut']).sum() / num_graphs
# per_equal = (results[results['case'] == 'ec']['maxcut'] == results[results['case'] == 'qaoa']['maxcut']).sum() / num_graphs
# pre_worse = (results[results['case'] == 'ec']['maxcut'] < results[results['case'] == 'qaoa']['maxcut']).sum() / num_graphs
#
# graphs_better = results[results['case'] == 'ec'][results[results['case'] == 'ec']['maxcut'] > results[results['case'] == 'qaoa']['maxcut']]['graph_index'].values
# graphs_equal = results[results['case'] == 'ec'][results[results['case'] == 'ec']['maxcut'] == results[results['case'] == 'qaoa']['maxcut']]['graph_index'].values
# graphs_worse = results[results['case'] == 'ec'][results[results['case'] == 'ec']['maxcut'] < results[results['case'] == 'qaoa']['maxcut']]['graph_index'].values
#
# print('Percentage of graphs where ec maxcut is better than qaoa')
# print(per_better, '\n', graphs_better)
# print('Percentage of graphs where ec maxcut is equal qaoa')
# print(per_equal, '\n', graphs_equal)
# print('Percentage of graphs where ec maxcut is worse than qaoa')
# print(pre_worse, '\n', graphs_worse)
#
# np.linalg.cond(matrix) #smaller the better
# nearest neighbor
#

#|%%--%%| <OjeNgwivjv|9U1tdTTr8C>


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # generate_graphs()
    # for g in range(11117):
    #     remove_max_degree_edge(g)
    #     print(f'g = {g}')
    # run_ec_parallel()


#|%%--%%| <9U1tdTTr8C|n0R2H72qsl>
r"""°°°
Plot results
°°°"""
#|%%--%%| <n0R2H72qsl|mIiYUVa7pQ>

# result_filename = f'/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/QAOA_dat.csv'
# df_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/Angle-Rounding-QAOA/result_analysis/qaoa.csv'
result_filename = f'/home/agwilkie/papers/angle_rounding/code/Angle-Rounding-QAOA/result_analysis/QAOA_dat.csv'
df_filename = '/home/agwilkie/papers/angle_rounding/code/Angle-Rounding-QAOA/result_analysis/qaoa.csv'

if os.path.exists(df_filename):
    qaoa_df = pd.read_csv(df_filename)
else:
    qaoa_df = pd.read_csv(result_filename)
    qaoa_df['graph_num'] = qaoa_df['path'].str.extract(r'graph_(\d+)')
    qaoa_df['graph_num'] = qaoa_df['graph_num'].astype(int)
    qaoa_df.set_index('graph_num', inplace=True)
    # qaoa_df['case'] = qaoa_df['random_path'].str.extract(r'(\w+).gml')
    qaoa_df.to_csv(df_filename, index=False)

results_2 = pd.read_csv('results_2.csv')
results_2['graph_num'] = results_2['path'].str.extract(r'graph_(\d+)')
results_2['graph_num'] = results_2['graph_num'].astype(int)
results_eval_2 = results_2[['graph_num', 'evals']].rename(columns={'evals': 'evals_2'})
results_eval_2.set_index('graph_num', inplace=True)
results_eval_2['evals_2'] = results_eval_2['evals_2'].apply(lambda x: -np.round(np.fromstring(x[1:-1], sep=' '), 3)[0])

results_3 = pd.read_csv('results_3.csv')
results_3['graph_num'] = results_3['path'].str.extract(r'graph_(\d+)')
results_3['graph_num'] = results_3['graph_num'].astype(int)
results_eval_3 = results_3[['graph_num', 'evals']].rename(columns={'evals': 'evals_3'})
results_eval_3.set_index('graph_num', inplace=True)
results_eval_3['evals_3'] = results_eval_3['evals_3'].apply(lambda x: -np.round(np.fromstring(x[1:-1], sep=' '), 3)[0])

results_4 = pd.read_csv('results_4.csv')
results_4['graph_num'] = results_4['path'].str.extract(r'graph_(\d+)')
results_4['graph_num'] = results_4['graph_num'].astype(int)
results_eval_4 = results_4[['graph_num', 'evals']].rename(columns={'evals': 'evals_4'})
results_eval_4.set_index('graph_num', inplace=True)
results_eval_4['evals_4'] = results_eval_4['evals_4'].apply(lambda x: -np.round(np.fromstring(x[1:-1], sep=' '), 3)[0])

results_5 = pd.read_csv('results_5.csv')
results_5['graph_num'] = results_5['path'].str.extract(r'graph_(\d+)')
results_5['graph_num'] = results_5['graph_num'].astype(int)
results_eval_5 = results_5[['graph_num', 'evals']].rename(columns={'evals': 'evals_5'})
results_eval_5.set_index('graph_num', inplace=True)
results_eval_5['evals_5'] = results_eval_5['evals_5'].apply(lambda x: -np.round(np.fromstring(x[1:-1], sep=' '), 3)[0])


results_evals = pd.concat([results_eval_2, results_eval_3, results_eval_4, results_eval_5, qaoa_df['C']], axis=1).dropna()
# prettify(results_evals.tail(50), row_limit=50)

#|%%--%%| <mIiYUVa7pQ|dM5zvHziDt>


def make_plot(results_evals, graph_num):
    # x axis is the number of basis states (2, 3, 4, 5)
    # y axis is the energy
    # horizontal line is the QAOA energy

    x = [2, 3, 4, 5]
    y = results_evals.loc[graph_num].values[:-1]
    COLOR = 'black'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    plt.plot(x, y, marker='o')
    plt.axhline(results_evals.loc[graph_num, 'C'], color='r')
    plt.xlabel('Number of basis states')
    plt.ylabel('Energy')
    plt.title(f'Graph {graph_num}', color=COLOR)
    plt.savefig(f'plots/graph_{graph_num}.png')
    plt.close()

    # plt.show()

for i in list(results_evals.index):
    make_plot(results_evals, i)
