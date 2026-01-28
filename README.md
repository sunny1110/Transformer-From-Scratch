# Transformer From Scratch

A vanilla translation transformer written from scratch in PyTorch, implementing the architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

## Features

- Complete transformer architecture (encoder-decoder)
- Multi-head self-attention
- Positional encoding
- English to Italian translation using the OPUS Books dataset

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies and set up the project
uv sync
```

## Usage

### Training

```bash
uv run train
```

### Configuration

Training parameters can be modified in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `num_epochs` | 1000 | Number of training epochs |
| `lr` | 1e-4 | Learning rate |
| `seq_len` | 350 | Maximum sequence length |
| `d_model` | 512 | Model dimension |
| `N` | 6 | Number of encoder/decoder layers |
| `h` | 8 | Number of attention heads |
| `dropout` | 0.1 | Dropout rate |
| `d_ff` | 2048 | Feed-forward dimension |

## Project Structure

```
├── src/
│   ├── config.py      # Training configuration
│   ├── dataset.py     # Dataset and data loading
│   ├── model.py       # Transformer architecture
│   └── train.py       # Training loop
├── experiments/       # Jupyter notebooks for experiments
├── paper/             # Original paper PDF
├── weights/           # Saved model checkpoints
└── runs/              # TensorBoard logs
```

## Monitoring

View training progress with TensorBoard:

```bash
tensorboard --logdir runs/
```
