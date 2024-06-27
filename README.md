# PQC for function evaluation 

***Preliminaries: Aim and Definitions***  

Consider an $n$-qubit *input* register, denoted with subscript $i$, as well as an $m$-qubit *target* register, denoted with subscript $t$, and let $f: \mathbb{R} \to \mathbb{R}$ be a function that can be efficiently computed. In the following, we aim to construct and operator that (approximately) performs 
$$ \ket{j}_i \ket{0}_t  \mapsto \ket{j}_i \ket{f(j)}_t,$$
where $\ket{j}_i$, $\ket{f(j)}_t$ are computational basis states of the input and target register, respectively, representing digital encodings of the real numbers $j$, $f(j)$.  

***Approach: a QCNN***

The approach taken to realise the aim outlined above is a *quantum convolutional neural network (QCNN)*. This is a parameteric quantum circuit (PQC) involving the input and target register that consists of several layers. The qubits in the input register only appear as controlling qubits and are not directly altered in the circuit. The following layers will appear in the network:

1. **Input layer**. This replaces the so-called feature map that typically is used as the first layer in a QCNN, which maps classical data to an appropriate quantum representation. The role of this input layer is to feed the state of the input register into the circuit, ensuring that the final state of the target register is linked to the state of the input register, as required for a function evaluation circuit. In principle, the input layer can involve any arrangement of gates applied to the target register, controlled by the input qubits.  

    As a first ansatz, we choose the following approach consisting of applying $CR_y$ rotations to the target qubits. First, apply Hadamard gates to all target qubits, initialising the target register into a symmetric superposition of computational basis states. Then apply a $CR_y(\theta_k)$ rotation to the $k$-th target qubit with the $k$-th input qubit acting as control and the angle $\theta_k$ consituting one of the parameters in the PQC. Note that in the case $n \neq m$ multiple rotation gates may be applied to a single target qubit each controlled by a different input qubit (if $n >m$) or alternatively (if $n < m$) some target qubits may have no rotation gates applied to them at all.  While this qubit-to-qubit mapping is a rather crude approach to feeding the input state into the target register, the convolutional layers of the QCNN ensure mixing of the various target qubits which should result in the input state sufficiently "diffusing" through the target register. 

    This input layer, as described above, requires $n$ gates (of which no $CX$ gates), and $n$ circuit parameters. This will be denoted as $N^G_I =n$ (counting rotation gates) and $N^\theta_I=n$. 

2. **Convolutional layer**. A convolutional layer involves the cascaded application of a two-qubit operator,$\hat{Q}$, (more on which later) to various combinations of qubits in the target register. Note that, in principle, an operator acting on more than two qubits could be used, although this will not be explored here (for now). The function of the convolutional layer is to set and "mix" the states of the individual target qubits. We will explore two different types of convolutional layers in the following. 
    1. A *linear* or *nearest-neighbour* layer, in which the two-qubit operator is applied to target qubits $k$ and $k+1$ for $k \in \{0,...,m-2\}$ as well as to target qubits $0$ and $m-1$. This requires $m = \mathcal{O}(m)$ applications od the two-qubit operator and results in a circular entanglement of neighbouring target qubits. 
    2. A *quadratic* or *all-to-all* layer, in which the two-qubit operator is applied to all possible two-qubit combinations in the register, requiring $\binom{m}{2} = \frac{1}{2}m (m-1) = \mathcal{O}(m^2)$ applications of the operator and resulting in full entanglement between all target qubits. 
    
    It is not clear *a priori* whether linear or quadratic layers are better suited for the purposes of the circuit. As an initial approach, alternating linear and quadratic layers will be used. Note that [Sim 2019](https://arxiv.org/pdf/1905.10876) might prove useful for exploring different architectures. Denoting by $N^{G}_Q$ the gate cost of the two-qubit operator $\hat{Q}$ and by $N^\theta_{Q}$ the number of free circuit parameters associated with the operator, the total gate cost and number of parameters for $L_{NN}$ nearest-neighbour layers and $L_{AA}$ all-to-all layers is 
    $$ N^G_{C} = L_{NN} m N^G_Q + L_{AA} \frac{1}{2} m (m-1) N^G_Q,$$      
    $$ N^\theta_{C} = L_{NN} m N^\theta_Q + L_{AA} \frac{1}{2} m (m-1) N^\theta_Q.$$

    The operator $\hat{Q}$, may in principle contain up to 15 free parameters (as it corresponds to a general $4 \times 4$ unitary matrix). In order to reduce the parameter space of the QCNN, an implementation for $\hat{Q}$ is chosen that involves only three free parameters:
    $$ \hat{Q} = \mathcal{N} (\alpha, \beta, \gamma) \equiv \exp \left( \alpha \sigma_x \otimes \sigma_x +  \beta \sigma_y \otimes \sigma_y +  \gamma \sigma_z \otimes \sigma_z \right),$$
    where the $\sigma_i$ represent the $2 \times 2$ Pauli matrices. This approach is taken from the Qiskit Machine learning documentation ([see here](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html)). As is shown in [Vatan 2004](https://journals.aps.org/pra/pdf/10.1103/PhysRevA.69.032315) the operator $\mathcal{N} (\alpha, \beta, \gamma)$ can be represented using three $CX$ and five rotation gates. Thus, we find $N^G_Q = 8$ (counting rotation gates) and $N^\theta_Q =3$. It remains to be seen whether this choice of two-qubit operator is overly restrictive and whether a more general operator is required for adequate circuit performance (or, for example, different convolutional layers using different operators). 

CNNs (and QCNNs) typically have another type of layer, which alternates with the convolutional layers, the *pooling layer*. The purpose of pooling is to reduce the number of output (qu)bits in the network, as (Q)CNNs are commonly used for binary classification tasks. For the present purposes, pooling layers are not required. Based on the performance of a circuit involving only an input layer followed by a series of convolutional layers, it might be worthwhile experimenting with additional input layers placed between convolutional layers in order to ensure sufficient information input from the input register. 