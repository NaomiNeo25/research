import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime.lime_text
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import visualization as viz
import logging
from logging.handlers import RotatingFileHandler

from torch import nn

from bpe import load_bpe_tokenizer
import seaborn as sns
from torch.utils.data import DataLoader
from dlp import TextDataset, collate_fn
import warnings

from finale.models import HierarchicalAttentionNetwork

warnings.filterwarnings('ignore')

class ModelVisualizer:
    def __init__(
            self,
            results_dir="./visualization_results",
            max_seq_length=512,
            embed_dim=300,
            device=None,
            num_samples=100
    ):
        self.results_dir = results_dir
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_samples = num_samples

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'lime'), exist_ok=True)

        # Set up logging
        self.logger = self._setup_logging()

        # Load BPE tokenizer
        self.tokenizer = load_bpe_tokenizer('bpe.model')

        self.logger.info(f'Using device: {self.device}')

    def _setup_logging(self):
        log_file = os.path.join(self.results_dir, "visualization.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, mode='a')
        handler.setFormatter(formatter)
        logger = logging.getLogger('ModelVisualizer')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def _prepare_model_and_data(self, model_class, model_path, dataset_folder):
        train_data = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
        test_data = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))

        unique_labels = sorted(train_data['author_id'].unique())
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        label_map_reverse = {idx: label for label, idx in label_map.items()}

        vocab_size = self.tokenizer.get_piece_size()
        num_classes = len(unique_labels)

        self.logger.info(f"Initializing {model_class.__name__} with:")
        self.logger.info(f"  vocab_size: {vocab_size}")
        self.logger.info(f"  embed_dim: {self.embed_dim}")
        self.logger.info(f"  num_classes: {num_classes}")

        # Create the embedding layer first as it's needed for HAN
        embedding_layer = nn.Embedding(vocab_size, self.embed_dim)


        if model_class.__name__ == "HierarchicalAttentionNetwork":
            model = model_class(
                embedding_layer=embedding_layer,
                num_classes=num_classes,
                word_hidden_dim=50,
                sent_hidden_dim=50,
                dropout=0.5,
                sent_length=20,
                max_sent_length=25
            ).to(self.device)
        else:
            model = model_class(
                vocab_size=vocab_size,
                embed_dim=self.embed_dim,
                num_classes=num_classes
            ).to(self.device)

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        sample_data = test_data.sample(n=min(self.num_samples, len(test_data)), random_state=42)
        texts = sample_data['text'].tolist()
        labels = [label_map[label] for label in sample_data['author_id']]

        return model, texts, labels, label_map_reverse

    def model_predict_proba(self, model, text):
        if isinstance(text, str):
            tokens = torch.tensor([self.tokenizer.encode(text)[:self.max_seq_length]]).to(self.device)
            lengths = torch.tensor([min(len(tokens[0]), self.max_seq_length)]).to(self.device)
        else:
            encoded = [self.tokenizer.encode(t)[:self.max_seq_length] for t in text]
            max_len = max(len(seq) for seq in encoded)
            padded = [seq + [0] * (max_len - len(seq)) for seq in encoded]
            tokens = torch.tensor(padded).to(self.device)
            lengths = torch.tensor([len(seq) for seq in encoded]).to(self.device)

        with torch.no_grad():
            outputs = model(tokens, lengths)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def visualize_lime(self, model_class, model_path, dataset_folder, dataset_name):
        self.logger.info("Generating LIME visualizations...")

        try:
            model, texts, labels, label_map_reverse = self._prepare_model_and_data(
                model_class, model_path, dataset_folder
            )

            def predict_proba_wrapper(texts):
                probs = [self.model_predict_proba(model, text) for text in texts]
                return np.array(probs).squeeze()

            explainer = lime.lime_text.LimeTextExplainer(
                class_names=[str(label_map_reverse[i]) for i in range(len(label_map_reverse))]
            )

            for i in range(min(5, len(texts))):
                self.logger.info(f"Processing text {i + 1}/5")
                self.logger.info(f"Text: {texts[i][:100]}...")
                self.logger.info(f"True label: {label_map_reverse[labels[i]]}")

                try:
                    exp = explainer.explain_instance(
                        texts[i],
                        predict_proba_wrapper,
                        num_features=20,
                        num_samples=100,
                        top_labels=1  # Only explain top prediction
                    )


                    pred_label = exp.available_labels()[0]

                    self.logger.info(f"Predicted label: {label_map_reverse[pred_label]}")

                    # Create visualization
                    plt.figure(figsize=(15, 8))
                    exp.as_pyplot_figure(label=pred_label)
                    plt.title(f'Explanation for class: {label_map_reverse[pred_label]}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.results_dir, 'lime', f'explanation_{dataset_name}_{i}.png'),
                                bbox_inches='tight', dpi=300)
                    plt.close()

                    # Save HTML version
                    html_path = os.path.join(self.results_dir, 'lime', f'explanation_{dataset_name}_{i}.html')
                    html_exp = exp.as_html()
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_exp)

                    self.logger.info(f"Successfully generated explanation for text {i + 1}")

                except Exception as e:
                    self.logger.error(f"Error processing text {i + 1}: {str(e)}")
                    self.logger.exception("Full traceback for this text:")
                    continue

            self.logger.info("LIME visualizations completed")

        except Exception as e:
            self.logger.error(f"Error in LIME visualization: {str(e)}")
            self.logger.exception("Full traceback:")

    def generate_all_visualizations(self, model_class, model_path, dataset_folder, dataset_name):
        self.logger.info(f"Generating visualizations for {dataset_name}")
        self.visualize_lime(model_class, model_path, dataset_folder, dataset_name)
        self.logger.info(f"All visualizations completed for {dataset_name}")