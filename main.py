import argparse 
from tools import train_QNN, test_QNN 

parser = argparse.ArgumentParser(usage='', description="Train and test the QCNN.")   
parser.add_argument('--n', help="Number of input qubits.", default=2, type=int)
parser.add_argument('--m', help="Number of target qubits.", default=2, type=int)
parser.add_argument('--L', help="Number of network layers. If multiple values given will execute sequentially.", default=[6],type=int, nargs="+")
parser.add_argument('--loss', help="Loss function.", default="CE", choices=["CE", "MSE", "L1", "KLD"])

parser.add_argument('--f', help="Function to evaluate (variable: x).", default="x")
parser.add_argument('--f_str', help="String describing function.", default="x")

parser.add_argument('--epochs', help="Number of epochs.", default=300,type=int)
parser.add_argument('--seed', help="Random seed.", default=1680458526,type=int)
parser.add_argument('--lr', help="Learning rate.", default=0.01,type=float)
parser.add_argument('--b1', help="Adam optimizer b1 parameter.", default=0.7,type=float)
parser.add_argument('--b2', help="Adam optimizer b2 parameter.", default=0.999,type=float)
parser.add_argument('--shots', help="Number of shots used by sampler.", default=300,type=int)
parser.add_argument('--meta', help="String with meta data.", default="")

opt = parser.parse_args()

for i in range(len(opt.L)):
    train_QNN(n=int(opt.n),m=int(opt.m),L=int(opt.L[i]), seed=int(opt.seed), shots=int(opt.shots), lr=float(opt.lr), b1=float(opt.b1), b2=float(opt.b2), epochs=int(opt.epochs), func=lambda x: eval(opt.f), func_str=opt.f_str, loss_str=opt.loss, meta=opt.meta)
    test_QNN(n=int(opt.n),m=int(opt.m),L=int(opt.L[i]),epochs=int(opt.epochs), func=lambda x: eval(opt.f), func_str=opt.f_str, loss_str=opt.loss, meta=opt.meta)


