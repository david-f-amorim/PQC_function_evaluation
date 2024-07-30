import numpy as np 
from tools import A_generate_network, psi  
from qiskit import QuantumCircuit,Aer, execute 
from qiskit.circuit import ParameterVector
from qiskit.quantum_info.operators import Operator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from torch import Tensor, no_grad
from torch.optim import Adam
import time, sys, os, warnings 
import torch

epochs= 600

# prepare circuit with amplitudes
n=6 
circuit = QuantumCircuit(n) 
qubits = list(range(n))
weights_A = np.load("ampl_outputs/weights_6_3_600_x76_MM_40_168_zeros.npy")  
circuit.compose(A_generate_network(n, 3), qubits, inplace=True)
circuit = circuit.assign_parameters(weights_A)

# construct parametrised circuit with U3 rotations 
param_index =0
params=ParameterVector("theta", 3*n)
for i in np.arange(n):
        par = params[int(param_index) : int(param_index + 3)]
        circuit.u(par[0],par[1],par[2], qubits[i])
        param_index +=3 

# construct operator from target amplitudes and phases 
x_min = 40
x_max = 168 
dx = (x_max-x_min)/(2**n) 
x_arr = np.arange(x_min, x_max, dx) 
ampl_target = x_arr**(-7./6)
ampl_target = ampl_target / np.sqrt(np.sum(ampl_target**2))
phase_target = psi(np.linspace(0, 2**n, len(x_arr)),mode="psi")

operator=np.zeros((2**n,2**n),dtype=complex)

for i in np.arange(2**n):
    for j in np.arange(2**n):
        if ampl_target[i]*ampl_target[j] != 0:
            operator[i,j]=1/(ampl_target[i]*ampl_target[j]) * np.exp(1.j * (phase_target[i] -phase_target[j]))

O=Operator(operator)

# wrap as EstimatorQNN 
qnn = EstimatorQNN(
                circuit=circuit.decompose(),            
                input_params=[],   
                weight_params=circuit.parameters,   
                observables=O 
            )  
initial_weights =np.zeros(len(circuit.parameters))
model = TorchConnector(qnn, initial_weights)

optimizer = Adam(model.parameters(), lr=0.01, betas=(0.7, 0.999), weight_decay=0.005, maximize=True) 
input=Tensor([])

loss_vals=np.empty(epochs)
mismatch_vals=np.empty(epochs)

# start training 
print(f"\n\nTraining started. Epochs: {epochs}. Input qubits: {n}. \n")
start = time.time() 

warnings.filterwarnings("ignore", category=np.ComplexWarning)

for i in np.arange(epochs):

        # train model  
        optimizer.zero_grad()
        loss = torch.abs(model(input)) / (2**(2*n))
        loss.backward()
        optimizer.step()

        # save loss for plotting 
        loss_vals[i]=loss.item()

        # get mismatch 
        qc= QuantumCircuit(n) 
        q = list(range(n))
        weights_A = np.load("ampl_outputs/weights_6_3_600_x76_MM_40_168_zeros.npy")  
        qc.compose(A_generate_network(n, 3), q, inplace=True)
        qc = qc.assign_parameters(weights_A)

        param_index =0
        pars=ParameterVector("theta_new", 3*n)
        for j in np.arange(n):
            par = pars[int(param_index) : int(param_index + 3)]
            qc.u(par[0],par[1],par[2], qubits[i])
            param_index +=3 

        with no_grad():
            generated_weights = model.weight.detach().numpy()
  
        qc = qc.assign_parameters(generated_weights)

        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        result = job.result()
        state_vector = np.asarray(result.get_statevector())

        fidelity=np.abs(np.dot(ampl_target * np.exp(1.j*phase_target),np.conjugate(state_vector)))**2
        mismatch = 1. - np.sqrt(fidelity) 
        mismatch_vals[i]=mismatch

        np.save(os.path.join("outputs", f"ESTIMATOR_state_vector_{epochs}"),state_vector)

        # print status
        a = int(20*(i+1)/epochs)
       
        if i==0:
            time_str="--:--:--.--"
        elif i==epochs-1:
            time_str="00:00:00.00"    
        else:
            remaining = ((time.time() - start) / i) * (epochs - i)
            mins, sec = divmod(remaining, 60)
            hours, mins = divmod(mins, 60)
            time_str = f"{int(hours):02}:{int(mins):02}:{sec:05.2f}"

        prefix="\t" 
        print(f"{prefix}[{u'█'*a}{('.'*(20-a))}] {100.*((i+1)/epochs):.2f}% ; Loss {loss_vals[i]:.2e} ; Mismatch {mismatch_vals[i]:.2e} ; ETA {time_str}", end='\r', file=sys.stdout, flush=True)

warnings.filterwarnings("default", category=np.ComplexWarning)        
        
print(" ", flush=True, file=sys.stdout)
    
elapsed = time.time()-start
mins, sec = divmod(elapsed, 60)
hours, mins = divmod(mins, 60)
time_str = f"{int(hours):02}:{int(mins):02}:{sec:05.2f}" 

# decompose circuit for gate count 
num_CX = dict(qc.decompose(reps=4).count_ops())["cx"]
num_gates = num_CX + dict(qc.decompose(reps=4).count_ops())["u"]

print(f"\nTraining completed in {time_str}. Number of weights: {len(generated_weights)}. Number of gates: {num_gates} (of which CX gates: {num_CX}). \n\n")

np.save(os.path.join("outputs", f"ESTIMATOR_weights_{epochs}"),generated_weights)
np.save(os.path.join("outputs", f"ESTIMATOR_mismatch_{epochs}"),mismatch_vals)
np.save(os.path.join("outputs", f"ESTIMATOR_loss_{epochs}"),loss_vals)
