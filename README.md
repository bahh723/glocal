# glocal

1.  Run run.sh, which runs the glocal algorithm on the isolet_reduced dataset. It is a 5-worker dataset derived from the original isolet dataset (https://archive.ics.uci.edu/ml/datasets/isolet), but each only having 100 samples. 

2.  Three methods are implemented (change them in main.py)
   
    Glocal: specified by Simulator.MODE_JOINT
   
    Central: Simulator.MODE_CENTRAL
   
    Independent: Simulator.MODE_INDEP
   
3.  For technical reasons, a tmp/ folder is created in the training time. 
