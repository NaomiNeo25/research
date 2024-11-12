from models import *
from visualize import ModelVisualizer
import os

def run_visualization():
    # Initialize visualizer
    visualizer = ModelVisualizer(
        results_dir="./visualization_results",
        num_samples=100,
        embed_dim=300,
        max_seq_length=512
    )

    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "experiment_results", "models", "best_model_IMDB62_HierarchicalAttentionNetwork_foldfull.pth"))
    dataset_folder = os.path.abspath(os.path.join(current_dir, "datasets", "IMDB62"))

    print(f"Model path: {model_path}")
    print(f"Dataset folder: {dataset_folder}")

    # Generate visualizations
    visualizer.generate_all_visualizations(
        model_class=HierarchicalAttentionNetwork,
        model_path=model_path,
        dataset_folder=dataset_folder,
        dataset_name='IMBD62'
    )

if __name__ == "__main__":
    run_visualization()