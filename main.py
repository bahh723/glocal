from vowpalwabbit import pyvw
import numpy as np
import argparse
import Simulator
from Simulator import Simulator
from DataReader import DataReader

parser = argparse.ArgumentParser()
parser.add_argument("--n_worker", type=int, default=5)
parser.add_argument("--lr", type=float)
parser.add_argument("--lr2", type=float, default=0)
parser.add_argument("--dataset", type=str)
parser.add_argument("--epoch", type=int, default=1)

args = parser.parse_args()

data_reader = DataReader(filename=args.dataset, n_worker=args.n_worker)
data, n_class = data_reader.read_dataset()
    
sim = Simulator(mode=Simulator.MODE_JOINT, dataset=data, n_worker=args.n_worker, lr=args.lr, lr2=args.lr2, n_class=n_class)

# ======== baseline methods =========
#sim = Simulator(mode=Simulator.MODE_INDEP, dataset=data, n_worker=n_worker, lr=args.lr, n_class=n_class)         
#sim = Simulator(mode=Simulator.MODE_CENTRAL, dataset=data, n_worker=n_worker, lr=args.lr, n_class=n_class) 
# ===================================

tr_loss, tr_iter = sim.run_simulation_classification(is_training=True)
