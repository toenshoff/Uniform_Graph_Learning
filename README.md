The installation and code was tested only on Linux (Ubuntu 22.04 LTS) with CUDA 11.7 installed.

Install Dependencies for GPU (with anaconda):
```
conda create --name env python=3.10
conda activate env
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2.0 -c pyg
pip install -r requirements.txt
```

To reproduce the experiments on Unbounded Countable Features (UCF) run:
```
bash scripts/run_ucf_experiments.sh
```

To reproduce the experiments on Single Value Features (SF) run:
```
bash scripts/run_svf_experiments.sh
```
