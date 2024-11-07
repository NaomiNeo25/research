import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import logging
from logging.handlers import RotatingFileHandler
from dlp import TextDataset, collate_fn
import seaborn as sns
import matplotlib.pyplot as plt
from bpe.bpe import load_bpe_tokenizer
import numpy as np


class ModelTester:
    def __init__(
            self,
            model_dir,
            results_dir="./test_results",
            batch_size=64,
            max_seq_length=512,
            device=None
    ):
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Set up logging
        self.logger = self._setup_logging()

        # Load BPE tokenizer
        self.tokenizer = load_bpe_tokenizer('bpe.model')

        self.logger.info(f'Using device: {self.device}')

    def _setup_logging(self):
        """Set up logging configuration"""
        log_file = os.path.join(self.results_dir, "testing.log")

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            mode='a'
        )
        handler.setFormatter(formatter)

        logger = logging.getLogger('ModelTester')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def plot_confusion_matrix(self, y_true, y_pred, labels, dataset_name, model_name):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Save plot
        plot_path = os.path.join(self.results_dir, f'confusion_matrix_{dataset_name}_{model_name}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        self.logger.info(f'Confusion matrix saved to {plot_path}')

    def evaluate_model(self, model, test_loader, label_map_reverse):
        """Evaluate model performance"""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for tokens, labels, lengths in test_loader:
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                outputs = model(tokens, lengths)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert numeric labels back to original labels
        original_preds = [label_map_reverse[pred] for pred in all_preds]
        original_labels = [label_map_reverse[label] for label in all_labels]

        return {
            'loss': total_loss / len(test_loader),
            'predictions': original_preds,
            'true_labels': original_labels,
            'probabilities': np.array(all_probs),
            'numeric_preds': all_preds,
            'numeric_labels': all_labels
        }

    def test_model(self, model_class, model_path, dataset_folder, dataset_name):
        """Test a specific model on a dataset"""
        self.logger.info(f'Testing {model_path} on {dataset_name}')

        try:
            # Load test data
            test_data = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))

            # Load training data to get label mapping
            train_data = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))

            # Create label mappings
            unique_labels = sorted(train_data['author_id'].unique())
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            label_map_reverse = {idx: label for label, idx in label_map.items()}

            # Prepare test data
            test_texts = test_data['text'].tolist()
            test_labels = [label_map[label] for label in test_data['author_id']
                           if label in label_map]

            # Create test dataset and dataloader
            test_dataset = TextDataset(test_texts, test_labels, self.tokenizer,
                                       max_length=self.max_seq_length)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                     shuffle=False, collate_fn=collate_fn)

            # Initialize and load model
            vocab_size = self.tokenizer.get_piece_size()
            num_classes = len(unique_labels)
            model = model_class(vocab_size=vocab_size, num_classes=num_classes).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))

            # Evaluate model
            results = self.evaluate_model(model, test_loader, label_map_reverse)

            # Calculate metrics
            test_f1 = f1_score(results['true_labels'], results['predictions'], average='weighted')

            # Generate and save detailed classification report
            report = classification_report(results['true_labels'], results['predictions'],
                                           target_names=[str(label) for label in unique_labels],
                                           digits=4)

            # Save results
            model_name = os.path.basename(model_path).replace('.pth', '')
            results_path = os.path.join(self.results_dir, f'results_{dataset_name}_{model_name}.txt')

            with open(results_path, 'w') as f:
                f.write(f'Model: {model_name}\n')
                f.write(f'Dataset: {dataset_name}\n')
                f.write(f'Test F1 Score: {test_f1:.4f}\n')
                f.write('\nClassification Report:\n')
                f.write(report)

            # Plot confusion matrix
            self.plot_confusion_matrix(
                results['numeric_labels'],
                results['numeric_preds'],
                unique_labels,
                dataset_name,
                model_name
            )

            self.logger.info(f'Results saved to {results_path}')
            self.logger.info(f'Test F1 Score: {test_f1:.4f}')

            return {
                'model_name': model_name,
                'dataset': dataset_name,
                'f1_score': test_f1,
                'results_path': results_path,
                'predictions': results['predictions'],
                'true_labels': results['true_labels'],
                'probabilities': results['probabilities']
            }

        except Exception as e:
            self.logger.error(f"Error testing model {model_path}: {str(e)}")
            self.logger.exception("Full traceback:")
            return None