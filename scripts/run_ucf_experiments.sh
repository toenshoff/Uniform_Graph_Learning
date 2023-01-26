#!/bin/bash

for i in {1..5}; do
  python train_ucf.py --aggr mean --num_layers 2 --u_num_layers 2 --hidden_dim 256 --lr 0.001 --seed ${i} --model_dir models/ucf/mean_s${i}
  python test_ucf.py --ckpt_path models/ucf/mean_s${i}/best.ckpt

  python train_ucf.py --aggr sum --num_layers 2 --u_num_layers 2 --hidden_dim 256 --lr 0.001 --seed ${i} --model_dir models/ucf/sum_s${i}
  python test_ucf.py --ckpt_path models/ucf/sum_s${i}/best.ckpt
done
