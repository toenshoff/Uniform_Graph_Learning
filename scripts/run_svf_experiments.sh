#!/bin/bash

for i in {1..5}; do
  python train_svf.py --aggr sum mean --num_layers 2 --u_num_layers 2 --hidden_dim 256 --lr 0.0001 --seed ${i} --model_dir models/svf/sum_mean_s${i}
  python test_svf.py --ckpt_path models/svf/sum_mean_s${i}/best.ckpt

  python train_svf.py --aggr sum --num_layers 2 --u_num_layers 2 --hidden_dim 256 --lr 0.0001 --seed ${i} --model_dir models/svf/sum_s${i}
  python test_svf.py --ckpt_path models/svf/sum_s${i}/best.ckpt
done
