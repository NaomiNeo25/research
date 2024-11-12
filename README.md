# Attention-Based Deep Learning for Generalizable Authorship Attribution Across Diverse Text Domains

This repository contains the code for the research project *Attention-Based Deep Learning for Generalizable Authorship Attribution Across Diverse Text Domains*. This project investigates the integration of attention mechanisms in neural networks to enhance authorship attribution performance across varied text domains.

## Requirements

To run the code, you will need the following dependencies:

- **Python** 3.12
- **PyTorch**
- **CUDA** 12.1 (for GPU support)
- **Pandas**
- **Scikit-learn (sklearn)**
- **NumPy**
- **Matplotlib**
- **LIME** (for interpretability)
- **SentencePiece** (for tokenization)
- **Captum** (for model interpretability)

Install dependencies with:
```bash
pip install torch pandas scikit-learn numpy matplotlib lime sentencepiece captum
```

## Folder Structure

The repository expects the following folder structure:

```
current_folder/
│
├── datasets/
│   ├── CCAT50/
│   │   └── *.csv
│   ├── Blog50/
│   │   └── *.csv
│   └── IMDb62/
│       └── *.csv
│
├── experiment_results/
│
└── visualization_results/
```

- **`datasets/`**: Contains subfolders named after each dataset (e.g., `CCAT50`, `Blog50`, `IMDb62`), with CSV files containing the data for each respective dataset.
- **`experiment_results/`**: Stores the results generated during experiments.
- **`visualization_results/`**: Contains visualizations and interpretability results.

## Running Experiments

1. **Prepare the Tokenizer**: Run the `train_bpe_tokenizer` method in `bpe.py` to generate the BPE tokenizer if it hasn’t been set up yet.

2. **Train Models**: To train a model on a specific dataset, use the training script corresponding to the dataset. For example:
   ```bash
   python train_[datasetname].py
   ```
   Replace `[datasetname]` with the name of the dataset you wish to train on (e.g., `CCAT50`, `Blog50`, `IMDb62`).

## Project Structure

- `bpe.py`: Script for training the BPE tokenizer.
- `train_[datasetname].py`: Script to train models on each dataset.
- `visualization_results/`: Stores visualizations for attention and interpretability.
