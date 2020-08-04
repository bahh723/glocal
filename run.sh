#!/bin/sh

mkdir tmp
dataset=isolet/isolet

python3 main.py --n_worker 5 \
                --lr 0.002 \
                --dataset ${dataset}


