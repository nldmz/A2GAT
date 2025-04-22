# A²GAT: Adaptive Anchor-based Graph Attention Networks

Implementation of Adaptive Anchor-based Graph Attention Networks for large-scale sparse bipartite graph embedding.

## Requirements

```bash
python>=3.8
torch>=1.8.0
numpy>=1.19.2
scikit-learn>=0.24.2
scipy>=1.6.2
```

## Directory Structure

```
.
├── dataset/              # Dataset directory
├── models/              # Saved model directory
├── train.py            # Training script for recommendation
├── train_lp.py         # Training script for link prediction
├── model.py            # Model architecture
├── utils.py            # Utility functions
└── calculate.py        # Anchor points calculation
```

## Data Format

Prepare your datasets in the following structure:

```
dataset/
└── [dataset_name]/
    ├── train.csr.pickle  # Training set (sparse matrix)
    ├── test.csr.pickle   # Test set (sparse matrix)
    ├── lp.train.npz      # Link prediction training set
    └── lp.test.npz       # Link prediction test set
```

## Running Instructions

### Recommendation Task

Basic training:
```bash
python train.py --dataset DBLP --gpu 0 --dim 64
```


### Link Prediction Task

Basic training:
```bash
python train_lp.py --data_root_dir ./dataset --model_file models/Wikipedia/model.pt --gpu 0
```
