#!/bin/bash
dataset=crime
loss=mmd


for lmda in 0.0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019 0.02; do
    python main.py --dataset $dataset --save_dir save/$loss/$dataset/ --constraint_loss_func $loss --mode variational --num_layer 2 --lmda_init $lmda --device 3 --variational_epochs 100 --step_size 0.001 --batch_size 1024
done