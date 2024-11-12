import logging
from logging.handlers import RotatingFileHandler
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
import random
import sentencepiece as spm
from bpe import train_bpe_tokenizer, load_bpe_tokenizer
from dlp import TextDataset, collate_fn


class Trainer:
    def __init__(
            self,
            save_dir="./experiment_results",
            batch_size=64,
            learning_rate=0.001,
            num_epochs=100,
            patience=5,
            reduce_lr_factor=0.5,
            l2_penalty=1e-5,
            embed_dim=300,
            hidden_dim=128,
            max_seq_length=512,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            seed=42
    ):
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.reduce_lr_factor = reduce_lr_factor
        self.l2_penalty = l2_penalty
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.seed = seed

        # Create directories
        self.model_base_dir = os.path.join(save_dir, "models")
        os.makedirs(self.model_base_dir, exist_ok=True)

        # logging
        self.logger = self._setup_logging()

        # random seeds
        self._set_seed()

        self.logger.info(f'Using device: {self.device}')

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _setup_logging(self):
        os.makedirs(self.save_dir, exist_ok=True)

        main_log_file = os.path.join(self.save_dir, "experiment.log")
        error_log_file = os.path.join(self.save_dir, "experiment_errors.log")

        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers = []

        for handler in [
            RotatingFileHandler(main_log_file, maxBytes=10 * 1024 * 1024, backupCount=5, mode='a'),
            logging.StreamHandler(),
            RotatingFileHandler(error_log_file, maxBytes=10 * 1024 * 1024, backupCount=5, mode='a')
        ]:
            handler.setFormatter(detailed_formatter)
            root_logger.addHandler(handler)

        root_logger.info("Logging system initialized")
        return root_logger

    def initialize_model(self, model_class, vocab_size, num_classes):
        embedding_layer = nn.Embedding(vocab_size, self.embed_dim)

        model_params = {
            'TextCNN': {
                'vocab_size': vocab_size,
                'embed_dim': self.embed_dim,
                'num_classes': num_classes
            },
            'BLSTM_TextCNN': {
                'embedding_layer': embedding_layer,
                'num_classes': num_classes
            },
            'ParallelCNNBLSTM': {
                'vocab_size': vocab_size,
                'embed_dim': self.embed_dim,
                'num_classes': num_classes,
                'hidden_dim': self.hidden_dim
            },
            'ParallelCNNBLSTMWithAttention': {
                'embedding_layer': embedding_layer,
                'num_classes': num_classes,
                'hidden_dim': self.hidden_dim
            },
            'ParallelCNNBLSTMWithPreConcatAttention': {
                'embedding_layer': embedding_layer,
                'num_classes': num_classes,
                'hidden_dim': self.hidden_dim
            },
            'ParallelCNNBLSTMWithMHA_BeforeConcat': {
                'embedding_layer': embedding_layer,
                'num_classes': num_classes,
                'hidden_dim': self.hidden_dim
            },
            'ParallelCNNBLSTMWithMHA_AfterConcat': {
                'embedding_layer': embedding_layer,
                'num_classes': num_classes,
                'hidden_dim': self.hidden_dim
            },
            'HierarchicalAttentionNetwork': {
                'embedding_layer': embedding_layer,
                'num_classes': num_classes,
                'word_hidden_dim': 50,
                'sent_hidden_dim': 50,
                'dropout': 0.5,
                'sent_length': 20,
                'max_sent_length': 25,
            }
        }

        if model_class.__name__ not in model_params:
            raise ValueError(f"Model {model_class.__name__} not supported")

        params = model_params[model_class.__name__]
        model = model_class(**params).to(self.device)
        return model

    def train_epoch(self, model, train_loader, criterion, optimizer):
        model.train()
        total_loss = 0

        for tokens, labels, lengths in train_loader:
            tokens = tokens.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

            optimizer.zero_grad()
            outputs = model(tokens, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate_model(self, model, data_loader):
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for tokens, labels, lengths in data_loader:
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                outputs = model(tokens, lengths)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1
        }

    def train_model(self, model, train_loader, val_loader, dataset_name, fold=None, model_name=None):
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty)
        criterion = nn.CrossEntropyLoss()
        best_metrics = None  # Initialize best_metrics at the start

        fold_str = f"_fold{fold}" if fold is not None else ""
        model_str = f"_{model_name}" if model_name is not None else ""
        best_model_path = os.path.join(
            self.model_base_dir,
            f'best_model_{dataset_name}{model_str}{fold_str}.pth'
        )

        if val_loader is not None:
            # Training with validation data
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=self.patience, factor=self.reduce_lr_factor
            )
            best_val_loss = float('inf')
            epochs_no_improve = 0
        else:
            # Training without validation data
            scheduler = None
            best_train_loss = float('inf')
            epochs_no_improve = 0
            patience = self.patience

        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0

            for tokens, labels, lengths in train_loader:
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                optimizer.zero_grad()
                outputs = model(tokens, lengths)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total

            if val_loader is not None:

                val_metrics = self.evaluate_model(model, val_loader)
                current_metrics = val_metrics

                # Log progress
                self.logger.info(
                    f'Epoch {epoch + 1}/{self.num_epochs}, '
                    f'Training Loss: {avg_train_loss:.4f}, '
                    f'Training Accuracy: {train_accuracy:.2f}%, '
                    f'Validation Loss: {val_metrics["loss"]:.4f}, '
                    f'Validation Accuracy: {val_metrics["accuracy"]:.2f}%, '
                    f'Validation F1: {val_metrics["f1_score"]:.4f}'
                )

                scheduler.step(val_metrics['loss'])

                # Check for improvement based on validation loss
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_metrics = val_metrics
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), best_model_path)
                    self.logger.info(
                        f'New best model saved with Validation Loss: {best_val_loss:.4f}, '
                        f'Accuracy: {val_metrics["accuracy"]:.2f}%, '
                        f'F1 Score: {val_metrics["f1_score"]:.4f}'
                    )
                else:
                    epochs_no_improve += 1
                    self.logger.info(f'No improvement for {epochs_no_improve} epoch(s).')
            else:

                current_metrics = self.evaluate_model(model, train_loader)  # Evaluate on training data

                # Log progress
                self.logger.info(
                    f'Epoch {epoch + 1}/{self.num_epochs}, '
                    f'Training Loss: {avg_train_loss:.4f}, '
                    f'Training Accuracy: {train_accuracy:.2f}%'
                )

                # Check for improvement based on training loss
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss
                    best_metrics = current_metrics
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), best_model_path)
                    self.logger.info(
                        f'New best model saved with Training Loss: {best_train_loss:.4f}, '
                        f'Training Accuracy: {train_accuracy:.2f}%'
                    )
                else:
                    epochs_no_improve += 1
                    self.logger.info(f'No improvement in training loss for {epochs_no_improve} epoch(s).')

            # Early stopping check for both cases
            if epochs_no_improve >= self.patience:
                self.logger.info("Early stopping triggered.")
                break

        # If we never improved, save the final model and metrics
        if best_metrics is None:
            self.logger.info("No improvement found during training, using final model state")
            torch.save(model.state_dict(), best_model_path)
            best_metrics = current_metrics

        return best_model_path, best_metrics

    def run_experiment(self, model_classes, dataset_folders):
        tokenizer = load_bpe_tokenizer()
        vocab_size = tokenizer.get_piece_size()

        self.logger.info(f'Vocabulary size: {vocab_size}')

        all_results = []  # Store results for all datasets

        for dataset_folder in dataset_folders:
            dataset_name = os.path.basename(dataset_folder)
            self.logger.info(f'\nProcessing dataset: {dataset_folder}')
            dataset_results = []  # Store results for current dataset

            try:
                train_data = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
                val_data = pd.read_csv(os.path.join(dataset_folder, 'validation.csv'))
                test_data = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))

                # Combine train and validation data for cross-validation
                combined_data = pd.concat([train_data, val_data], ignore_index=True)
                texts = combined_data['text'].tolist()
                labels = combined_data['author_id'].tolist()

                # Create dataset
                dataset = TextDataset(texts, labels, tokenizer, max_length=self.max_seq_length)

                # Get vocabulary size and number of classes
                vocab_size = tokenizer.get_piece_size()
                label_set = sorted(list(set(labels)))
                label_map = {label: idx for idx, label in enumerate(label_set)}
                num_classes = len(label_set)
                labels_mapped = [label_map[label] for label in labels]

                # Prepare test data
                test_texts = test_data['text'].tolist()
                test_labels = [label_map.get(label, -1) for label in test_data['author_id'].tolist()]
                test_samples = [(text, label) for text, label in zip(test_texts, test_labels) if label != -1]

                if not test_samples:
                    self.logger.warning('No valid samples in the test set after filtering unseen labels.')
                    continue

                test_texts, test_labels = zip(*test_samples)
                test_dataset = TextDataset(list(test_texts), list(test_labels), tokenizer,
                                           max_length=self.max_seq_length)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

                # 5-Fold Cross-Validation
                kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)

                for model_class in model_classes:
                    model_name = model_class.__name__
                    self.logger.info(f'\nTraining {model_name} on {dataset_name}')

                    fold_metrics = []
                    for fold, (train_indices, val_indices) in enumerate(kf.split(texts, labels_mapped), 1):
                        self.logger.info(f'\nFold {fold}')


                        train_subset = Subset(dataset, train_indices)
                        val_subset = Subset(dataset, val_indices)

                        train_loader = DataLoader(
                            train_subset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            collate_fn=collate_fn
                        )
                        val_loader = DataLoader(
                            val_subset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            collate_fn=collate_fn
                        )


                        model = self.initialize_model(model_class, vocab_size, num_classes)
                        best_model_path, best_metrics = self.train_model(
                            model, train_loader, val_loader, dataset_name, fold, model_name
                        )
                        fold_metrics.append(best_metrics)

                    # Calculate average metrics across folds
                    avg_val_acc = np.mean([m['accuracy'] for m in fold_metrics])
                    avg_val_f1 = np.mean([m['f1_score'] for m in fold_metrics])

                    # Retrain model on full training data
                    all_train_indices = np.arange(len(dataset))
                    full_train_subset = Subset(dataset, all_train_indices)
                    full_train_loader = DataLoader(
                        full_train_subset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        collate_fn=collate_fn
                    )

                    model = self.initialize_model(model_class, vocab_size, num_classes)
                    best_model_path, _ = self.train_model(
                        model, full_train_loader, None, dataset_name, 'full', model_name
                    )

                    # Evaluate on test set
                    model.load_state_dict(torch.load(best_model_path))
                    test_metrics = self.evaluate_model(model, test_loader)

                    # Record results
                    result = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "avg_val_accuracy": avg_val_acc,
                        "avg_val_f1": avg_val_f1,
                        "test_accuracy": test_metrics['accuracy'],
                        "test_f1": test_metrics['f1_score'],
                        "test_loss": test_metrics['loss']
                    }
                    dataset_results.append(result)
                    all_results.append(result)

                    self.logger.info(
                        f'{model_name} on {dataset_name}:\n'
                        f'Average Validation Accuracy: {avg_val_acc:.2f}%\n'
                        f'Average Validation F1: {avg_val_f1:.4f}\n'
                        f'Test Accuracy: {test_metrics["accuracy"]:.2f}%\n'
                        f'Test F1: {test_metrics["f1_score"]:.4f}'
                    )



                dataset_results_df = pd.DataFrame(dataset_results)

                dataset_results_path = os.path.join(
                    self.save_dir,
                    f"experiment_results_{dataset_name}.csv"

                )

                dataset_results_df.to_csv(dataset_results_path, index=False)

                self.logger.info(f"Results for {dataset_name} saved to {dataset_results_path}")

            except Exception as e:
                self.logger.error(f"Error processing dataset {dataset_folder}: {str(e)}")
                self.logger.exception("Full traceback:")
                continue


        all_results_df = pd.DataFrame(all_results)
        combined_results_path = os.path.join(self.save_dir, f"experiment_results_{dataset_name}.csv")
        all_results_df.to_csv(combined_results_path, index=False)
        self.logger.info(f"Combined results for all datasets saved to {combined_results_path}")
        return all_results

