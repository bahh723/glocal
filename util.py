import numpy as np

    
def concat_label_feature(label=None, feature=None, base=None): 
    if label==None:
        return "| " + feature
    elif base!=None: 
        return str(label) + " 1 " + str(base) + " | " + feature 
    else: 
        return str(label) + " | " + feature


