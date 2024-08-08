"""
### *pqcprep* provides parametrised quantum circuits (PQCs) for quantum state preparation. 

The aim of *pqcprep* is to implement the algorithm for quantum state preparation described in [Background](#background) via gate-efficient parametrised quantum circuits (PQCs), as described in [Approach](#approach). 
However, the functionality provided as part of the package is general enough to be adapted to a wide range of other applications. 

# Usage 

*pqcprep* provides an out-of-the-box command-line tool...

# Approach

The key challenge tackled by *pqcprep* is to construct a PQC that can perform function evaluation: $\ket{j}\ket{0} \mapsto \ket{j} \ket{\Psi'(j)}$, for some analytical function 
$\Psi$, with $\Psi ' \equiv \Psi / 2 \pi$. Throughout this documentation, the $n$-qubit register containing the $\ket{j}$ and the $m$-qubit register containing the $\ket{\Psi'(j)}$ 
will be referred to as the "input register" and "target register", respectively.

A quantum convolutional neural network (QCNN) is used to approach the problem. A QCNN is a parametrised quantum circuit involving multiple layers. Two types of network layers 
are implemented here:

- convolutional layers (CL) involve multi-qubit entanglement gates; 

- input layers (IL) (replacing the conventional QCNN pooling layers) involve controlled single-qubit operations on target qubits. 

Input qubits only appear as controls throughout the QCNN. 

### Convolutional Layers 

Each CL involves the cascaded application of a two-qubit operator on the target register. A general two-qubit operator involves 15 parameters. Hence, to reduce the parameter space, 
the canonical three-parameter operator

$$
\\mathcal{N}(\\alpha, \\beta, \\gamma) =  \exp \\left( i \\left[ \\alpha X \otimes X + \\beta Y \\otimes Y + \\gamma Z \otimes Z \\right] \\right)
$$

is applied, at the cost of restricting the search space. This can be decomposed (see [Vatan 2004](https://arxiv.org/pdf/quant-ph/0308006)) into 3 CX, 3 $\\text{R}_\\text{z}$, and 2 $\\text{R}_\\text{y}$ gates.
A two-parameter real version, $\\mathcal{N}_\\mathbb{R}(\lambda, \mu)$, can be obtained by removing the $\\text{R}_\\text{z}$. 

Two convolutional layer topologies are implemented, loosely based on [Sim 2019](https://arxiv.org/pdf/1905.10876): 

- neighbour-to-neighbour/linear CLs: the $\\mathcal{N}$ (or $\\mathcal{N}_\\mathbb{R}$) gate is applied to neighbouring target qubits; 

- all-to-all/quadratic CLs: the $\\mathcal{N}$ (or $\\mathcal{N}_\\mathbb{R}$) gate is applied to all combinations of target qubits. 

The $\mathcal{N}$-gate cost of neighbour-to-neighbour (NN) layers is $\\mathcal{O}(m)$ while that of all-to-all (AA) layers is $\\mathcal{O}(m^2)$.
The QCNN uses alternating linear and quadratic CLs.

### Input Layers 

ILs, replacing pooling layers, feed information about the input register into the target register.  
An IL involves a sequence of controlled generic single-qubit rotations (CU3 gates) on the target qubits, with input qubits as controls.
For an IL producing states with real amplitudes, the CU3 gates are replaced with $\\text{CR}_\\text{y}$ gates.
Each input qubit controls precisely one CU3 (or $\\text{CR}_\\text{y}$ operation), resulting in an $\\mathcal{O}(n)$ gate cost.
ILs are inserted after every second convolutional layer, alternating between control states 0 and 1. 

### Training the QCNN 

For training, the QCNN is wrapped as a [SamplerQNN](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.neural_networks.SamplerQNN.html) 
object and connected to PyTorch's [Adam optimiser](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) via [TorchConnector](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.connectors.TorchConnector.html). 
The optimiser determines improved parameter values for each training run ("epoch") based on the calculated loss between output and target state. 
Beyond loss, mismatch is an important metric:
$$
M= 1 - |\\braket{\\psi_\\text{target}| \\psi_\\text{out}}|. 
$$

There are two ways to train the QCNN on input data:

1.  Training on individual states: one of the $2^n$ input states, $\ket{j},$ is randomly chosen each epoch. 
    The network is thus taught to transform $\ket{j}\ket{0} \mapsto \ket{j}\ket{\Psi'(j)} $ for each of the states individually.
2.  Training in superposition: the network is taught to transform 
$$
\\left(\\sum^{2^n-1}_{j=0} c_j \\ket{j} \\right) \\ket{0} \\mapsto  \\sum^{2^n -1}_{j=0} c_j \\ket{j}\\ket{\\Psi'(j)},
$$
    where the coefficients $c_j \\sim \\frac{1}{\\sqrt{2^n}}$ are randomly sampled each epoch.  
    By linearity, this teaches the network to transform $\ket{j}\ket{0} \mapsto \ket{j}\ket{\Psi'(j)} $ for each $\ket{j}$. 


One can also train the QCNN to produce a target distribution independent of the input register. This is equivalent to constructing an operator $\hat{U}_A$
such that 
$$\hat{U}_A \ket{0}^{\otimes n} =  \\frac{1}{|\\tilde{A}|} \sum^{2^n-1}_{j=0} \\tilde{A}(j) \ket{j}$$ 
for some distribution function $\\tilde{A}$.

# Background

*pqcprep* builds on a scheme for quantum state preparation presented in [Hayes 2023](https://arxiv.org/pdf/2306.11073):
a complex vector $\\boldsymbol{h} =\\lbrace \\tilde{A}_j e^{i \Psi (j)} | 0 \leq j < N \\rbrace$, where $\\tilde{A}$, $\Psi$ are real functions 
that can be computed efficiently, is prepared as the quantum state 
$$ \ket{h} = \\frac{1}{\\vert \\tilde{A} \\vert} \sum^{2^n-1}_{j=0} \\tilde{A}(j) e^{i \Psi (j)} \ket{j}, $$
using $n =\\lceil \log_2 N \\rceil$ qubits. 

This requires operators $\hat{U}_A$ and $\hat{U}_\Psi$ such that 
\\begin{align}
\hat{U}_A \ket{0}^{\otimes n} &=  \\frac{1}{|\\tilde{A}|} \sum^{2^n-1}_{j=0} \\tilde{A}(j) \ket{j}, \\newline
\hat{U}_\Psi \ket{j} &= e^{i \Psi (j)} \ket{j}.
\\end{align}

$\hat{U}_\Psi$ is constructed via an operator $\hat{Q}_\Psi$ that performs function evaluation in an ancilla register,
\\begin{equation}
\hat{Q}_\Psi  \ket{j} \ket{0}^{\otimes m}_a = \ket{j} \ket{\Psi'(j)}_a,
\\end{equation}
with $\Psi'(j) \equiv \Psi(j) / 2 \pi$, as well as an operator $\hat{R}$ that extracts the phase, 
$$ \hat{R} \ket{j} \ket{\Psi'(j)}_a = \ket{j} e^{i 2 \pi \Psi' (j)} \ket{\Psi' (j)}_a.$$
Thus, $\hat{U}_\Psi = \hat{Q}_{\Psi}^\dagger \hat{R} \hat{Q}_\Psi$ with $\hat{Q}_{\Psi}^\dagger$ clearing the ancilla register: 
\\begin{align} 
\hat{U}_\Psi  \hat{U}_A \ket{0} \ket{0}_a &= \\frac{1}{|\\tilde{A}|} \sum^{2^n-1}_{j=0} \\tilde{A}(j) \hat{U}_\Psi \ket{j} \ket{0}_a  \\newline 
&= \\frac{1}{|\\tilde{A}|} \sum^{2^n-1}_{j=0} \\tilde{A}(j)  \hat{Q}_\Psi^\dagger \hat{R} \hat{Q}_\Psi \ket{j} \ket{0}_a  \\newline 
&= \\frac{1}{|\\tilde{A}|} \sum^{2^n-1}_{j=0} \\tilde{A}(j)  \hat{Q}_\Psi^\dagger \hat{R}  \ket{j} \ket{\Psi'(j)}_a  \\newline 
&= \\frac{1}{|\\tilde{A}|} \sum^{2^n-1}_{j=0} \\tilde{A}(j)  \hat{Q}_\Psi^\dagger \ket{j} e^{i \Psi(j)} \ket{\Psi'(j)}_a  \\newline 
&= \\frac{1}{|\\tilde{A}|} \sum^{2^n-1}_{j=0} \\tilde{A}(j) e^{i \Psi(j)} \ket{j}  \ket{0}_a  \\newline 
&= \ket{h} \ket{0}_a  \\newline 
\\end{align}

This size, $m$, of the ancilla register limits the precision to which $\Psi(j)$ can be encoded to $\sim 2^{1-m} \pi$. 

# Imprint 


David Amorim, 2024. Email: [*2538354a@student.gla.ac.uk*](mailto:2538354a@student.gla.ac.uk) .


This project was funded by a Carnegie Vacation Scholarship and supervised by Prof Sarah Croke (University of Glasgow, School of Physics and Astronomy). 

"""

DIR = "pqcprep" 
""" @private """

import os
if not os.path.isdir(os.path.join(DIR, "outputs")):
    os.mkdir(os.path.join(DIR, "outputs"))
if not os.path.isdir(os.path.join(DIR, "ampl_outputs")):
    os.mkdir(os.path.join(DIR, "ampl_outputs"))    