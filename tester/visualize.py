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
from bpe.bpe import load_bpe_tokenizer
import seaborn as sns
from torch.utils.data import DataLoader
from dlp import TextDataset, collate_fn
import warnings

warnings.filterwarnings('ignore')


class ModelVisualizer:
    def __init__(
            self,
            results_dir="./visualization_results",
            max_seq_length=512,
            device=None,
            num_samples=100  # Number of samples to use for SHAP/LIME analysis
    ):
        self.results_dir = results_dir
        self.max_seq_length = max_seq_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_samples = num_samples

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'shap'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'lime'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'integrated_gradients'), exist_ok=True)

        # Set up logging
        self.logger = self._setup_logging()

        # Load BPE tokenizer
        self.tokenizer = load_bpe_tokenizer('bpe.model')

        self.logger.info(f'Using device: {self.device}')

    def _setup_logging(self):
        """Set up logging configuration"""
        log_file = os.path.join(self.results_dir, "visualization.log")

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

        logger = logging.getLogger('ModelVisualizer')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _prepare_model_and_data(self, model_class, model_path, dataset_folder):
        """Prepare model and data for visualization"""
        # Load training data to get label mapping
        train_data = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
        test_data = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))

        # Create label mappings
        unique_labels = sorted(train_data['author_id'].unique())
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        label_map_reverse = {idx: label for label, idx in label_map.items()}

        # Initialize and load model
        vocab_size = self.tokenizer.get_piece_size()
        num_classes = len(unique_labels)
        model = model_class(vocab_size=vocab_size, num_classes=num_classes).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        # Prepare sample data
        sample_data = test_data.sample(n=min(self.num_samples, len(test_data)), random_state=42)
        texts = sample_data['text'].tolist()
        labels = [label_map[label] for label in sample_data['author_id'] if label in label_map]

        return model, texts, labels, label_map_reverse

    def model_predict_proba(self, model, text):
        """Wrapper for model prediction with probability output"""
        tokens = torch.tensor([self.tokenizer.encode(text)[:self.max_seq_length]]).to(self.device)
        lengths = torch.tensor([min(len(tokens[0]), self.max_seq_length)]).to(self.device)

        with torch.no_grad():
            outputs = model(tokens, lengths)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def visualize_shap(self, model_class, model_path, dataset_folder, dataset_name):
        """Generate SHAP visualizations"""
        self.logger.info("Generating SHAP visualizations...")

        try:
            model, texts, labels, label_map_reverse = self._prepare_model_and_data(
                model_class, model_path, dataset_folder
            )

            # Create SHAP explainer
            explainer = shap.Explainer(
                lambda x: np.array([self.model_predict_proba(model, text) for text in x]),
                texts[:self.num_samples]
            )

            # Calculate SHAP values
            shap_values = explainer(texts[:self.num_samples])

            # Generate and save visualizations
            plt.figure(figsize=(15, 10))
            shap.summary_plot(
                shap_values,
                texts[:self.num_samples],
                class_names=[str(label_map_reverse[i]) for i in range(len(label_map_reverse))],
                show=False
            )
            plt.savefig(os.path.join(self.results_dir, 'shap', f'summary_plot_{dataset_name}.png'))
            plt.close()

            # Generate individual explanations for a few samples
            for i in range(min(5, len(texts))):
                plt.figure(figsize=(15, 5))
                shap.plots.text(shap_values[i], show=False)
                plt.savefig(os.path.join(self.results_dir, 'shap', f'text_explanation_{dataset_name}_{i}.png'))
                plt.close()

            self.logger.info("SHAP visualizations completed")

        except Exception as e:
            self.logger.error(f"Error in SHAP visualization: {str(e)}")
            self.logger.exception("Full traceback:")

    def visualize_lime(self, model_class, model_path, dataset_folder, dataset_name):
        """Generate LIME visualizations"""
        self.logger.info("Generating LIME visualizations...")

        try:
            model, texts, labels, label_map_reverse = self._prepare_model_and_data(
                model_class, model_path, dataset_folder
            )

            # Initialize LIME explainer
            explainer = lime.lime_text.LimeTextExplainer(
                class_names=[str(label_map_reverse[i]) for i in range(len(label_map_reverse))]
            )

            # Generate explanations for a few samples
            for i in range(min(5, len(texts))):
                exp = explainer.explain_instance(
                    texts[i],
                    lambda x: np.array([self.model_predict_proba(model, text) for text in x]),
                    num_features=20,
                    num_samples=100
                )

                # Save visualization
                plt.figure(figsize=(15, 8))
                exp.as_pyplot_figure()
                plt.savefig(os.path.join(self.results_dir, 'lime', f'explanation_{dataset_name}_{i}.png'))
                plt.close()

                # Save HTML visualization
                html_path = os.path.join(self.results_dir, 'lime', f'explanation_{dataset_name}_{i}.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(exp.as_html())

            self.logger.info("LIME visualizations completed")

        except Exception as e:
            self.logger.error(f"Error in LIME visualization: {str(e)}")
            self.logger.exception("Full traceback:")

    def visualize_integrated_gradients(self, model_class, model_path, dataset_folder, dataset_name):
        """Generate Integrated Gradients visualizations"""
        self.logger.info("Generating Integrated Gradients visualizations...")

        try:
            model, texts, labels, label_map_reverse = self._prepare_model_and_data(
                model_class, model_path, dataset_folder
            )

            # Create IntegratedGradients instance
            ig = IntegratedGradients(model)

            for i in range(min(5, len(texts))):
                # Prepare input
                tokens = torch.tensor([self.tokenizer.encode(texts[i])[:self.max_seq_length]]).to(self.device)
                lengths = torch.tensor([min(len(tokens[0]), self.max_seq_length)]).to(self.device)

                # Calculate attributions
                attributions, delta = ig.attribute(
                    tokens,
                    target=labels[i],
                    return_convergence_delta=True,
                    internal_batch_size=1
                )

                # Convert tokens to words for visualization
                words = [self.tokenizer.decode([token.item()]) for token in tokens[0]]

                # Visualize attributions
                plt.figure(figsize=(15, 5))
                viz.visualize_text_attr(
                    attributions[0].cpu().numpy(),
                    words,
                    show_plot=False
                )
                plt.savefig(os.path.join(
                    self.results_dir,
                    'integrated_gradients',
                    f'explanation_{dataset_name}_{i}.png'
                ))
                plt.close()

            self.logger.info("Integrated Gradients visualizations completed")

        except Exception as e:
            self.logger.error(f"Error in Integrated Gradients visualization: {str(e)}")
            self.logger.exception("Full traceback:")

    def visualize_attention_weights(self, model, text, dataset_name):
        """Visualize attention weights if model has attention mechanism"""
        if hasattr(model, 'get_attention_weights'):
            try:
                # Get attention weights
                tokens = torch.tensor([self.tokenizer.encode(text)[:self.max_seq_length]]).to(self.device)
                lengths = torch.tensor([min(len(tokens[0]), self.max_seq_length)]).to(self.device)

                attention_weights = model.get_attention_weights(tokens, lengths)

                # Visualize attention weights
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    attention_weights[0].cpu().numpy(),
                    cmap='viridis',
                    xticklabels=[self.tokenizer.decode([token.item()]) for token in tokens[0]],
                    yticklabels=False
                )
                plt.title('Attention Weights Visualization')
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f'attention_weights_{dataset_name}.png'))
                plt.close()

            except Exception as e:
                self.logger.error(f"Error in attention visualization: {str(e)}")
                self.logger.exception("Full traceback:")

    def generate_all_visualizations(self, model_class, model_path, dataset_folder, dataset_name):
        """Generate all types of visualizations"""
        self.logger.info(f"Generating visualizations for {dataset_name}")

        # Generate SHAP visualizations
        self.visualize_shap(model_class, model_path, dataset_folder, dataset_name)

        # Generate LIME visualizations
        self.visualize_lime(model_class, model_path, dataset_folder, dataset_name)

        # Generate Integrated Gradients visualizations
        self.visualize_integrated_gradients(model_class, model_path, dataset_folder, dataset_name)

        # Load model for attention visualization
        model = model_class(
            vocab_size=self.tokenizer.get_piece_size(),
            num_classes=len(pd.read_csv(os.path.join(dataset_folder, 'train.csv'))['author_id'].unique())
        ).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        # Generate attention visualizations if applicable
        sample_text = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))['text'].iloc[0]
        self.visualize_attention_weights(model, sample_text, dataset_name)

        self.logger.info(f"All visualizations completed for {dataset_name}")