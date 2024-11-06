import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from DLP import get_dataloaders, Reuters50Dataset, IMDb62Dataset, GuardianDataset, Blog50Dataset
from models import *
import numpy as np
import pandas as pd
import os
import sentencepiece as spm
import fasttext
import fasttext.util

# Set up directories
SAVE_DIR = "./experiment_results/BlogsDataset/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Set up logging
log_file = os.path.join(SAVE_DIR, "experiment.log")
error_log_file = os.path.join(SAVE_DIR, "experiment_errors.log")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file, mode='w'),
                        logging.StreamHandler()  # Outputs to console as well
                    ])

error_logger = logging.getLogger("error_logger")
error_handler = logging.FileHandler(error_log_file, mode='w')
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 20
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
DROPOUT_RATES = {"embed": 0.5, "blstm": 0.3, "penultimate": 0.2}
L2_PENALTY = 1e-5
HIDDEN_DIM = 128
EMBEDDING_DIM = 300
DEVICE = torch.device("cpu")

# Load BPE model
sp = spm.SentencePieceProcessor()
sp.Load('./bpe/bpe.model')
bpe_vocab_size = sp.piece_size()

# Initialize FastText embeddings
fasttext_model = fasttext.load_model('cc.en.300.bin')  # Assuming English FastText embeddings

# Define function to load FastText embeddings as torch embeddings
def get_fasttext_embeddings(vocab_size, embedding_dim):
    embedding = nn.Embedding(vocab_size, embedding_dim)
    for i in range(vocab_size):
        word = sp.id_to_piece(i)
        embedding.weight.data[i] = torch.tensor(fasttext_model[word]).float() if word in fasttext_model else torch.zeros(embedding_dim)
    return embedding

# Define function to load BPE embeddings
def get_bpe_embeddings(vocab_size, embedding_dim):
    return nn.Embedding(vocab_size, embedding_dim)

# Function for initializing a model
def initialize_model(model_class, embedding, num_classes):
    return model_class(embedding_layer=embedding, num_classes=num_classes).to(DEVICE)

# Function for training a model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    step = 0  # Initialize step counter
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for inputs, labels, lengths in train_loader:
            inputs, labels, lengths = inputs.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update total loss
            total_loss += loss.item()

            # Predictions and labels for this batch
            _, preds = torch.max(outputs, 1)
            batch_preds = preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()

            # Update overall predictions and labels
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

            # Calculate batch accuracy
            batch_acc = accuracy_score(batch_labels, batch_preds)
            # Optional: Calculate batch F1-score
            # batch_f1 = f1_score(batch_labels, batch_preds, average="weighted")

            # Log after each batch
            logging.info(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}], "
                f"Step [{step + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}, "
                f"Batch Accuracy: {batch_acc:.4f}"
                # f", Batch F1: {batch_f1:.4f}"  # Uncomment if you wish to log batch F1-score
            )

            # Update step counter
            step += 1

        # Scheduler step (you can decide whether to step per batch or per epoch)
        scheduler.step(total_loss)

        # Epoch-level metrics
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average="weighted")

        # Validation
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion)

        logging.info(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Completed. "
            f"Training Loss: {total_loss / len(train_loader):.4f}, "
            f"Training Accuracy: {epoch_acc:.4f}, "
            f"Training F1: {epoch_f1:.4f}"
        )
        logging.info(
            f"Validation Loss: {val_loss:.4f}, "
            f"Validation Accuracy: {val_acc:.4f}, "
            f"Validation F1: {val_f1:.4f}"
        )

        # Reset step counter after each epoch
        step = 0

    return model

# Function for evaluating a model
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels, lengths in data_loader:
            inputs, labels, lengths = inputs.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(data_loader), acc, f1

# Main Experiment Loop
def run_experiment(model_classes, datasets, embedding_type="FastText"):
    results = []
    for dataset_name, dataset_info in datasets.items():
        dataset_class = dataset_info["class"]
        dataset_args = dataset_info["args"]

        # Prepare Dataloaders
        logging.info(f"\nRunning on dataset: {dataset_name} with {embedding_type} embeddings")
        try:
            # Load full dataset to access num_classes
            full_dataset = dataset_class(**dataset_args)
            num_classes = full_dataset.num_classes  # Access num_classes from the original dataset

            # Split into train, validation, and test loaders
            train_loader, val_loader, test_loader = get_dataloaders(dataset_class, dataset_args, BATCH_SIZE)

            # Select embedding type
            vocab_size = bpe_vocab_size  # Using BPE vocabulary size for both embeddings
            if embedding_type == "FastText":
                embedding = get_fasttext_embeddings(vocab_size, EMBEDDING_DIM)
            else:
                embedding = get_bpe_embeddings(vocab_size, EMBEDDING_DIM)

            for model_class in model_classes:
                # Initialize model and optimizer
                logging.info(f"Training model: {model_class.__name__} on {dataset_name} with {num_classes} classes using {embedding_type} embeddings")
                model = initialize_model(model_class, embedding, num_classes=num_classes)
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_PENALTY)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=REDUCE_LR_PATIENCE, factor=REDUCE_LR_FACTOR)
                criterion = nn.CrossEntropyLoss()

                # Train and Evaluate
                model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
                test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, criterion)

                # Save model
                model_save_path = os.path.join(SAVE_DIR, f"{dataset_name}_{model_class.__name__}_{embedding_type}.pt")
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"Model saved to {model_save_path}")

                # Log results
                results.append({
                    "dataset": dataset_name,
                    "model": model_class.__name__,
                    "embedding_type": embedding_type,
                    "num_classes": num_classes,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "test_f1": test_f1
                })
        except Exception as e:
            error_logger.error(f"Error in dataset {dataset_name} with embedding {embedding_type}: {str(e)}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(SAVE_DIR, f"experiment_results_{embedding_type}.csv")
    results_df.to_csv(results_csv_path, index=False)
    logging.info(f"Results saved to {results_csv_path}")
    return results

# Define datasets and models with root directories/files
datasets = {
    "Blogs": {"class": Blog50Dataset, "args": {"root_dir": './datasets/blogs', "sp_model": sp, "max_seq_length": 512}}
}

#Check if all models work one by one
model_classes = [
    BLSTM2DCNN,
]

# model_classes = [
#     BLSTM2DCNN, BLSTM2DCNNWithAttention, ParallelBLSTMCNNWithAttention, HAN, TextCNN, BLSTM2DCNNWithMultiHeadAttention, RCNN
# ]

# Run experiments first with FastText, then with BPE embeddings
logging.info("Starting experiments with FastText embeddings...")
results_fasttext = run_experiment(model_classes, datasets, embedding_type="FastText")

logging.info("\nStarting experiments with BPE embeddings...")
results_bpe = run_experiment(model_classes, datasets, embedding_type="BPE")
