"""
### *pqcprep* provides parametrised quantum circuits (PQCs) for quantum state preparation. 

The aim of *pqcprep* is to implement the algorithm for quantum state preparation described in [Background](#background) via gate-efficient parametrised quantum circuits (PQCs), as described in [Approach](#approach). 
However, the functionality provided as part of the package is general enough to be adapted to a wide range of other applications. 

# Usage 

*pqcprep* provides an out-of-the-box command-line tool to construct and train PQCs for quantum state preparation as well as visualisaing their performance. Additionally, the package provides 
a range of functions to be used for more bespoke applications. 

## Installation 

*pqcprep* is published on [PYPI](https://pypi.org), the python publishing index. Thus, it can be installed using the command 
                
                python3 -m pip install pqcprep 

This requires `pip`, which should be automatically installed when using python in most cases (see [here](https://pypi.org/project/pip/) for more information). 


The source code for *pqcprep* can also be downloaded from GitHub by cloning the directory via 

                git clone https://github.com/david-f-amorim/PQC_function_evaluation.git 

## Command-line Tool 

*pqcprep* comes with a built-in command-line tool. When installing *pqcprep* via `pip` (recommended) using the tool is as simple as running the command

                pqcprep [-OPTIONS]

When using *pqcprep* via a GitHub clone the command-line tool can be accessed by running 

                python3 -m pqcprep [-OPTIONS]

when in the same working directory as the downloaded directory `pqcprep`. 


The central function of the command-line tool is to construct a PQC for either function evaluation (phase encoding) or amplitude preparation, train it and
evaluate its performance. This is implemented by executing functions such as `pqcprep.training_tools.train_QNN()`, `pqcprep.training_tools.test_QNN()`, `pqcprep.training_tools.ampl_train_QNN()`,
`pqcprep.plotting_tools.benchmark_plots()` and `pqcprep.plotting_tools.benchmark_plots_ampl()`. A collection of different directories containing data files and plots is generated in the process. 
As the program is running, relevant information is displayed in the terminal. The most important output produced are the PQC weights which can be used to replicate the trained PQC and 
include it in various algorithms or other applications. 

### Options

The following options can be passed to the command-line tool. Note that many of these options are closely linked to the arguments of  `pqcprep.training_tools.train_QNN()`, `pqcprep.training_tools.test_QNN()`, `pqcprep.training_tools.ampl_train_QNN()`. 
Consulting the documentation for these functions can thus be useful.The order in which options are passed is irrelevant. Most options can be passed using a short-hand (often just one letter) beginning with `-` as well as a longer, more descriptive name beginning with `--`. 


Options requiring one (or more) variable(s) to be passed alongside them, i.e. `-option VAL` : 

- **-D**, **--DIR** :

    Set the parent directory for the output files: `-D X` or `--D` X sets the parent directory to `X`. If not provided, the current working directory is taken as the default.

- **-n**, **--n** : 

    Set number of input qubits: `-n X` or `--n X` sets the number of input qubits to `X`. If not provided, a default value of 6 is taken. 

- **-m**, **--m** : 

    Set number of target qubits: `-m X` or `--m X` sets the number of target qubits to `X`. If not provided, a default value of 3 is taken. This has no effect if `--A` or `--ampl` is passed. 

-  **-L**, **--L** :   

    Set number of network layers: `-L X` or `--L X` sets the number of network layers to `X`. Multiple options can be given and will be executed sequentially, e.g. `-L X Y Z` will 
    create three PQCs with `X`, `Y` and `Z` network layers, respectively.  If not provided, a default value of 6 is taken. 

- **-f**, **--f** : 

    Set phase function to be evaluated: `-f X` or `--f X` sets the phase function to `X`, where `X` must be one of the options for `mode` in `pqcprep.psi_tools.psi()`. This has no effect if `--A` or `--ampl` is passed. 
    If not provided, the default `'psi'` function is evaluated. 

- **-l**, **--loss** : 

    Set loss function to be used by the optimiser: `-l X` or `--loss X` sets the loss function to `X`, where `X` must be one of the optiosn for `loss_str` in `pqcprep.training_tools.set_loss_func()`. 
    If not provided, the default loss function is `SAM`. 

- **-e**, **--epochs** : 

    Set the number of epochs (training runs): `-e X` or `--epochs X` sets the number of epochs to `X`. If not provided, the default number of epochs is 600.

- **-M**, **--meta** :    

    Include a meta information in the file names of the output: `-M X` or `--meta X` results in the string `X` to be included in the name string created via 
    `pqcprep.file_tools.vars_to_name_str()` (or via `pqcprep.file_tools.vars_to_name_str_ampl()` if `-A` or `--ampl` is passed). Note that the string `X` may not 
    contain spaces or the sequence `'--'`. If not provided, no meta information is added to the file names. 

- **-ni**, **--nint** : 

    Set number of integer input qubits: `-ni X` or `--nint X` sets the number of integer input qubits to `X`. If not provided, all input qubits are taken to be integer qubits. 

- **-mi**, **--mint** : 

    Set number of integer target qubits: `-mi X` or `--mint X` sets the number of integer target qubits to `X`. If not provided, all target qubits are taken to be integer qubits. This has no effect if `--A` or `--ampl` is passed.     

-  **-d**, **--delta** :   

    Set value of the `delta` parameter used to train the function evaluation network in some contexts: `-d X` or `--delta X` sets the value of `delta` to `X`. Note that `X` must be a float between 0 and 1. Multiple options can be given and will be executed sequentially, e.g. `-d X Y Z` will 
    train three PQCs with `delta` equal to `X`, `Y` and `Z`, respectively. This has no effect unless `-TS` or `-H` are passed and if `-A` or `--ampl` are passed. If not provided, a default value of 0 is taken.     
    See `pqcprep.training_tools.train_QNN()` for more information. 

- **-p**, **--WILL_p** :

    Set value of the `p` parameter used by the `WILL` loss function: `-p X` or `--WILL_p X` sets the value of `p`  to `X`. Multiple options can be given and will be executed sequentially, e.g. `-p X Y Z` will 
    train three PQCs with `p` equal to `X`, `Y` and `Z`, respectively. This has no effect unless `-l WILL` is passed (i.e. the `WILL` los function is used). If not provided, a default value of 1 is used. 

- **-q**, **--WILL_q** :

    Set value of the `q` parameter used by the `WILL` loss function: `-q X` or `--WILL_q X` sets the value of `q`  to `X`. Multiple options can be given and will be executed sequentially, e.g. `-q X Y Z` will 
    train three PQCs with `q` equal to `X`, `Y` and `Z`, respectively. This has no effect unless `-l WILL` is passed (i.e. the `WILL` los function is used). If not provided, a default value of 1 is used. 

- **-fA**, **--f_ampl** :   

    Set amplitude function to be prepared: `-fA X` or `--f_ampl X` sets the amplitude function to `X`, where `X` must be one of the options for `mode` in `pqcprep.psi_tools.A()`. This has no effect unless `--A` or `--ampl` is passed. 
    If not provided, the default `'x76'` function is evaluated. 

- **--xmin** :   

    Set minimum of the amplitude function domain: `--xmin X` sets the mininum of the amplitude function domain to `X`. This has no effect unless `--A` or `--ampl` is passed. 
    If not provided, the default value of 40.0 is taken.    

- **--xmax** :   

    Set minimum of the amplitude function domain: `--xmax X` sets the maximum of the amplitude function domain to `X`. This has no effect unless `--A` or `--ampl` is passed. 
    If not provided, the default value of 168.0 is taken.       

- **--seed** : 

    Set the seed for random number generation: `--seed X` sets the seed to `X`. If not provided, the default value of 1680458526 is taken. 
    

Options not requiring values to be passed alongside them, i.e. `-option` : 

- **-h**, **--help** : 

    Show a summary of the various options and exit. 


### Examples 




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
