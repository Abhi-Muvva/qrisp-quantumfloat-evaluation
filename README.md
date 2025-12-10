# QuantumFloat vs Angle/Amplitude Encoding — A Practical Evaluation on Iris (Qrisp + Qiskit)

This repository contains a complete experimental pipeline comparing three classical-to-quantum data-encoding strategies:

- QuantumFloat encoding (via the Qrisp framework)
- Angle encoding
- Amplitude encoding

All methods are evaluated on the Iris classification task, using both centroid-based NN and k-NN-style quantum classifiers.
The goal is to measure runtime, resource usage, accuracy, and scalability, and to expose the practical limitations of Qrisp’s QuantumFloat arithmetic compared to standard Qiskit pipelines.
