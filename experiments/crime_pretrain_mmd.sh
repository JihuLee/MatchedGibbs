#!/bin/bash
dataset=crime
loss=mmd
pre_epoch=50

for pretrain_seed in 2025 2026 2027 2028 2029; do
    python main.py --dataset $dataset --mode pretrain --epochs_pre $pre_epoch --save_dir save/$loss/$dataset/ --lmda_grid 0.0 10.0 11 --constraint_loss_func $loss --pretrain_seed $pretrain_seed
done