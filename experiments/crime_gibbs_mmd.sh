#!/bin/bash
dataset=crime

num_burn=10
step=1e-3
permute_size=10
num_burn=5
pre_epoch=50

for lmda in 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0; do
    python main.py --dataset $dataset --save_dir save/mmd/$dataset/ --constraint_loss_func mmd --mode gibbs --lmda_init $lmda --epochs_pre $pre_epoch --device 0 --step_size $step --num_burn $num_burn --permute_size $permute_size
done