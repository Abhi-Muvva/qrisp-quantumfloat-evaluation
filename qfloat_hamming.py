"""
Shared utilities for QuantumFloat-based Hamming distance experiments
on the Iris dataset.

This module provides:
    - load_iris_two_features_scaled:
        Loads Iris petal length/width and scales them to [0, 1].

    - hamming_distance_trainpoint:
        QuantumFloat-based Hamming distance between two 2D integer vectors.

    - estimate_resources_per_distance_call:
        Builds a representative distance circuit for a given precision msize
        and returns gate-count / depth / CNOT-equivalent resource metrics.
"""

from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from qrisp import QuantumFloat, QuantumBool, cx


def load_iris_two_features_scaled() -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Load the Iris dataset, keep petal length & width, and scale them to [0, 1].

    Returns:
        X_scaled:
            Array of shape (150, 2) with scaled features
            [petal length, petal width].

        y:
            Array of shape (150,) with class labels in {0, 1, 2}.

        feature_names:
            List with the two feature names as strings.

        target_names:
            Array with the class names as strings.
    """
    iris = load_iris()
    X_full = iris.data[:, 2:4]  # [petal length, petal width]
    y_full = iris.target

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_full)

    feature_names = iris.feature_names[2:4]
    target_names = iris.target_names

    return X_scaled, y_full, feature_names, target_names


def hamming_distance_trainpoint(
    test_vec_int: np.ndarray,
    train_vec_int: np.ndarray,
    msize: int,
    shots: int = 1,
) -> Tuple[int, Dict[str, float]]:
    """
    Compute the Hamming distance between two 2D integer-encoded feature vectors
    using a QuantumFloat-based circuit.

    Each vector encodes [feature_1, feature_2] with each scalar in
    [0, 2^msize - 1]. The function:

        - Allocates QuantumFloat registers for test and train features.
        - Allocates a QuantumFloat distance accumulator.
        - Encodes classical integers into the QuantumFloats.
        - For each bit index b, creates a mismatch flag as XOR(test_bit, train_bit)
          and conditionally increments the distance register if the mismatch is 1.
        - Measures the distance QuantumFloat and returns the most likely outcome.

    Args:
        test_vec_int:
            NumPy array of shape (2,) with integer-encoded test features.

        train_vec_int:
            NumPy array of shape (2,) with integer-encoded train features.

        msize:
            Number of bits used to represent each scalar feature.

        shots:
            Number of measurement shots for the distance register.

    Returns:
        d_ham:
            Measured Hamming distance as an integer.

        meta:
            Dictionary with timing and qubit metrics:
                {
                    "encoding_time":   time spent on encoding the integers,
                    "encoding_qubits": logical data qubits for the four scalars,
                    "total_time":      total wall-clock time (build + run),
                    "total_qubits":    number of qubits in the session
                                       after circuit construction,
                }
    """
    if test_vec_int.shape != (2,) or train_vec_int.shape != (2,):
        raise ValueError("test_vec_int and train_vec_int must be 2-element integer vectors.")

    t_total_start = perf_counter()

    # QuantumFloat registers for test and train features
    T_f1 = QuantumFloat(msize, exponent=0, signed=False, name="T_f1")
    T_f2 = QuantumFloat(msize, exponent=0, signed=False, name="T_f2")
    X_f1 = QuantumFloat(msize, exponent=0, signed=False, name="X_f1")
    X_f2 = QuantumFloat(msize, exponent=0, signed=False, name="X_f2")

    max_mismatches = 2 * msize
    dist_bits = int(np.ceil(np.log2(max_mismatches + 1)))
    dist_bits = max(dist_bits, 1)

    dist = QuantumFloat(dist_bits, exponent=0, signed=False, name="dist")
    qs = dist.qs

    t_enc_start = perf_counter()

    T_f1[:] = int(test_vec_int[0])
    T_f2[:] = int(test_vec_int[1])
    X_f1[:] = int(train_vec_int[0])
    X_f2[:] = int(train_vec_int[1])
    dist[:] = 0

    t_enc_end = perf_counter()
    encoding_time = t_enc_end - t_enc_start

    # Logical data bits used for the four scalar features
    encoding_qubits = float(4 * msize)

    def add_mismatch(bit_a, bit_b, dist_qf, label: str) -> None:
        """
        Create a mismatch flag as XOR(bit_a, bit_b) and, if the flag is 1,
        increment the distance register by one.
        """
        mismatch = QuantumBool(name=f"mismatch_{label}")
        cx(bit_a, mismatch)
        cx(bit_b, mismatch)
        with mismatch:
            dist_qf += 1

    for b in range(msize):
        add_mismatch(T_f1[b], X_f1[b], dist, f"f1_b{b}")
        add_mismatch(T_f2[b], X_f2[b], dist, f"f2_b{b}")

    mes_results = dist.get_measurement(shots=shots)
    if len(mes_results) == 0:
        d_ham = 0
    else:
        d_val = max(mes_results.items(), key=lambda kv: kv[1])[0]
        d_ham = int(d_val)

    # Total qubits present in the session after building the circuit
    total_qubits = float(len(qs.qubits))

    t_total_end = perf_counter()
    total_time = t_total_end - t_total_start

    meta: Dict[str, float] = {
        "encoding_time": float(encoding_time),
        "encoding_qubits": float(encoding_qubits),
        "total_time": float(total_time),
        "total_qubits": float(total_qubits),
    }

    return d_ham, meta


def estimate_resources_per_distance_call(msize: int) -> Dict[str, float]:
    """
    Build and compile a representative Hamming-distance circuit for a single
    distance computation between two 2D integer-encoded points and extract
    resource estimates.

    The representative circuit uses dummy data but the same structure as
    hamming_distance_trainpoint, so its gate counts, depth, and qubit numbers
    are indicative of the cost per distance call for this precision.

    Args:
        msize:
            Number of bits per scalar feature.

    Returns:
        Dictionary with fields:
            - msize
            - num_qubits
            - depth
            - gate_count
            - num_1q_gates
            - num_2q_gates
            - num_3q_gates
            - num_gt3q_gates
            - num_mq_gates
            - cnot_equiv_per_distance_call
    """
    scale = (2 ** msize) - 1
    vec_a_int = np.array([scale // 3, scale // 2], dtype=int)
    vec_b_int = np.array([scale // 4, scale // 2], dtype=int)

    A_f1 = QuantumFloat(msize, exponent=0, signed=False, name="A_f1_res")
    A_f2 = QuantumFloat(msize, exponent=0, signed=False, name="A_f2_res")
    B_f1 = QuantumFloat(msize, exponent=0, signed=False, name="B_f1_res")
    B_f2 = QuantumFloat(msize, exponent=0, signed=False, name="B_f2_res")

    max_mismatches = 2 * msize
    dist_bits = int(np.ceil(np.log2(max_mismatches + 1)))
    dist_bits = max(dist_bits, 1)

    dist = QuantumFloat(dist_bits, exponent=0, signed=False, name="dist_res")
    qs = dist.qs

    A_f1[:] = int(vec_a_int[0])
    A_f2[:] = int(vec_a_int[1])
    B_f1[:] = int(vec_b_int[0])
    B_f2[:] = int(vec_b_int[1])
    dist[:] = 0

    def add_mismatch(bit_a, bit_b, dist_qf, label: str) -> None:
        mismatch = QuantumBool(name=f"mismatch_res_{label}")
        cx(bit_a, mismatch)
        cx(bit_b, mismatch)
        with mismatch:
            dist_qf += 1

    for b in range(msize):
        add_mismatch(A_f1[b], B_f1[b], dist, f"f1_b{b}")
        add_mismatch(A_f2[b], B_f2[b], dist, f"f2_b{b}")

    qc = qs.compile()

    gate_count_dic = qc.count_ops()
    total_gates = float(sum(gate_count_dic.values()))
    depth = float(qc.depth())
    num_qubits = float(len(qc.qubits))

    one_qubit_gates = {
        "x", "y", "z", "h", "s", "sdg", "t", "tdg",
        "rx", "ry", "rz", "u", "u1", "u2", "u3",
        "p", "id", "sx", "sxdg",
    }

    two_qubit_gates = {
        "cx", "cz", "swap", "iswap", "ecr",
        "rxx", "ryy", "rzz", "rzx",
        "QFT no swap",
        "QFT no swap_dg",
    }

    three_qubit_gates = {"ccx", "ccz", "cswap"}

    gt3_qubit_gates = {
        "mcx", "mcx_gray", "mcx_recursive", "mcx_vchain",
        "mcp", "mcu3", "mcrx", "mcry", "mcrz", "mcphase",
    }

    num_1q = 0.0
    num_2q = 0.0
    num_3q = 0.0
    num_gt3q = 0.0

    for name, cnt in gate_count_dic.items():
        c = float(cnt)
        if name in one_qubit_gates:
            num_1q += c
        elif name in two_qubit_gates:
            num_2q += c
        elif name in three_qubit_gates:
            num_3q += c
        elif name in gt3_qubit_gates:
            num_gt3q += c
        else:
            num_2q += c

    num_mq = num_3q + num_gt3q
    cnot_equiv = num_2q + 6.0 * num_3q + 10.0 * num_gt3q

    return {
        "msize": float(msize),
        "num_qubits": num_qubits,
        "depth": depth,
        "gate_count": total_gates,
        "num_1q_gates": num_1q,
        "num_2q_gates": num_2q,
        "num_3q_gates": num_3q,
        "num_gt3q_gates": num_gt3q,
        "num_mq_gates": num_mq,
        "cnot_equiv_per_distance_call": cnot_equiv,
    }
