import numpy as np
import random
import json

class DataReader:
   def __init__(self, filename,  n_worker):
      self.n_worker = n_worker
      self.filename = filename

   def read_dataset(self):  
      dataset = [[] for _ in range(self.n_worker)] 
      min_label = 10000
      max_label = -10000

      for i in range(self.n_worker):
         fn = self.filename + "_" + str(self.n_worker) + "_" + str(i)
         
         f = open(fn)
         for line in f: 
            line = line.strip()
            linesp = line.split(" ",1)
            label = int(linesp[0])
            if label < min_label: 
                min_label = label
            if label > max_label: 
                max_label = label
            feat = linesp[1]

            feat_global = ""
            feat_local = ""
            feat_split = feat.split()
            for entry in feat_split:
               entry_split = entry.split(":")
               feat_global += entry + " "
               feat_local += entry + " "

            dataset[i].append((feat_global, feat_local, label))

      return dataset, max_label-min_label+1


