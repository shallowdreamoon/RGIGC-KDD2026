# RGIGC

This repository contains the source code for the KDD 2026 paper:

**Reinforced Structural Reasoning for Receptive Field Optimization in GNN toward Interpretable Graph Clustering**

RGIGC optimizes the receptive field of graph neural networks for interpretable graph clustering by jointly reasoning over neighbor selection and aggregation depth.

## Repository Structure

```text
RGIGC/
  train_Cora.py              Main training script on Cora
  config.py                  Default hyperparameters
  gnn_env.py                 Reinforcement-learning environment
  gnn_model_1.py             GNN model
  multi_armed_bandit_v1.py   UCB-based policy optimization
  pretrain/                  Pretraining code
  pretrain_res/              Pretrained model checkpoint
  data/                      Cora data files
  results/                   Saved model/result files
```

## Requirements

The code requires Python and the following main packages:

```text
numpy
scipy
scikit-learn
torch
torch-geometric
gym
matplotlib
seaborn
```

Please install PyTorch and PyTorch Geometric versions compatible with your local CUDA/CPU environment.

## Run

From the code directory:

```bash
cd RGIGC
python train_Cora.py
```

The script runs RGIGC on the Cora dataset using the included data and pretrained checkpoint.

To regenerate the pretrained checkpoint:

```bash
cd RGIGC/pretrain
python pre_train_cora.py
```



