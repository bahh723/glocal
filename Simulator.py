from util import concat_label_feature
import numpy as np
from vowpalwabbit import pyvw
from time import time


class Simulator:
    MODE_INDEP=0
    MODE_CENTRAL=1
    MODE_JOINT=2

    def __init__(self, mode, dataset, n_worker, lr, lr2=0, n_class=0):
        self.mode = mode
        self.dataset = dataset
        self.n_worker = n_worker
        self.lr = lr
        self.lr2 = lr2
        self.vw_master = []
        self.vw_workers = []
        self.f_master = []
        self.f_worker_list = []
        self.n_class = n_class

    def run_simulation_classification(self, is_training=True, output_every=100):
        n_class = self.n_class 
        if is_training:
            for n in range(n_class): 
                if self.mode != Simulator.MODE_JOINT:
                    self.vw_master.append(pyvw.vw("--quiet --adaptive --learning_rate {}".format(self.lr)))
                    self.vw_workers.append([pyvw.vw("--quiet --adaptive --learning_rate {}".format(self.lr)) for _ in range(self.n_worker)])
                else: 
                    prefix = str(n) + str(time())
                    file_m = "tmp/master" + prefix
                    self.vw_master.append(pyvw.vw("--quiet --adaptive -r {} --learning_rate {} ".format(file_m, self.lr)))
                    #==== read one line ====
                    self.vw_master[n].predict("")
                    self.f_master.append(open(file_m))
                    self.f_master[n].readline()
                    #=======================
                    self.vw_workers.append([])
                    self.f_worker_list.append([])
                    for j in range(self.n_worker):
                        file_w = "tmp/worker" + prefix + str(j)
                        self.vw_workers[n].append(pyvw.vw("--quiet --adaptive -r {} --learning_rate {}".format(file_w, self.lr2)))
                        self.vw_workers[n][j].predict("")
                        self.f_worker_list[n].append(open(file_w))
                        self.f_worker_list[n][j].readline()
       
        iteration_list = np.zeros((self.n_worker,))
        for j in range(self.n_worker):
            iteration_list[j] = len(self.dataset[j])
        loss_list = np.zeros((self.n_worker, int(np.max(iteration_list))))

        for i in range(int(np.max(iteration_list))):
            for j in range(self.n_worker): 
                n_data = int(iteration_list[j])
                if i >= n_data:
                    continue
              
                x_gb, x_lc, y  = self.dataset[j][i]
               
                if self.mode==Simulator.MODE_INDEP:
                    scores = []
                    for n in range(n_class): 
                        scores.append( self.vw_workers[n][j].predict(concat_label_feature(feature=x_lc)) )
                    yhat = np.argmax(scores)

                    if y==yhat: loss_list[j,i] = 0
                    else: loss_list[j,i] = 1

                    if is_training: 
                        for n in range(n_class):  
                            if y==n:
                                self.vw_workers[n][j].learn(concat_label_feature(label=1, feature=x_lc))
                            else: 
                                self.vw_workers[n][j].learn(concat_label_feature(label=0, feature=x_lc))

    
                elif self.mode==Simulator.MODE_CENTRAL:
                    scores = []
                    for n in range(n_class):
                       scores.append( self.vw_master[n].predict(concat_label_feature(feature=x_gb)) )

                    yhat = np.argmax(scores)
                    if y==yhat: loss_list[j,i]=0
                    else: loss_list[j,i]=1
    
                    if is_training:
                        for n in range(n_class): 
                            if y==n: 
                               self.vw_master[n].learn(concat_label_feature(label=1, feature=x_gb))
                            else:
                               self.vw_master[n].learn(concat_label_feature(label=0, feature=x_gb))
    
                elif self.mode==Simulator.MODE_JOINT:
                    scores_m = []
                    scores_w = []
                    for n in range(n_class): 
                       self.vw_master[n].predict(concat_label_feature(feature=x_gb))
                       scores_m.append( float(self.f_master[n].readline()) )

                       self.vw_workers[n][j].predict(concat_label_feature(feature=x_lc))
                       scores_w.append( float(self.f_worker_list[n][j].readline()) )

                    scores = [scores_m[n] + scores_w[n] for n in range(n_class)]

                    yhat = np.argmax(scores) 
                    if y==yhat: loss_list[j,i]=0
                    else: loss_list[j,i]=1

                    if is_training: 
                       # learn
                       for n in range(n_class):  
                             base_m = scores_w[n]
                             if n==y:
                                self.vw_master[n].learn(concat_label_feature(label=1, feature=x_gb, base=base_m))
                             else:
                                self.vw_master[n].learn(concat_label_feature(label=0, feature=x_gb, base=base_m))
                             self.f_master[n].readline()
    
                       for n in range(n_class):
                             base_w = scores_m[n]
                             if n==y: 
                                self.vw_workers[n][j].learn(concat_label_feature(label=1, feature=x_lc, base=base_w))
                             else: 
                                self.vw_workers[n][j].learn(concat_label_feature(label=0, feature=x_lc, base=base_w))
                             self.f_worker_list[n][j].readline()
                
            if i % output_every == 0 and i>0:
                avg_loss_list = []
                for j in range(self.n_worker): 
                    n_data = int(iteration_list[j])
                    if i >= n_data:
                        continue
                    avg_loss_list.append(np.mean(loss_list[j,0:i]))
                print("iteration=", i , "/", int(max(iteration_list)), "   average cummulative error=", np.mean(avg_loss_list))
                  
        return loss_list, iteration_list

  
