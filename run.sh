#!/bin/sh

mkdir tmp
dataset=isolet_reduced/isolet

python3 main.py --n_worker 5 \
                --lr 0.002 \
                --lr 0.0005 \
                --dataset ${dataset}


