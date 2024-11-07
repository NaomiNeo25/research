
from models import *
from tester.visualize import ModelVisualizer


def run_visualization():
    # Initialize visualizer
    visualizer = ModelVisualizer(
        results_dir="./visualization_results",
        num_samples=100  # Number of samples to use for visualization
    )

    # Define visualization configs
    viz_configs = [
        {
            'model_class': BLSTM_CNN,
            'model_path': './experiment_results/models/best_model_blogs_fold5_BLSTM_CNN.pth',
            'dataset_folder': './datasets/blogs',
            'dataset_name': 'blogs'
        },
        {
            'model_class': HAN,
            'model_path': './experiment_results/models/best_model_Guardian_fold5_HAN.pth',
            'dataset_folder': './datasets/Guardian/Guardian_original',
            'dataset_name': 'Guardian'
        },
        # Add more configurations as needed
    ]

    # Generate visualizations
    for config in viz_configs:
        visualizer.generate_all_visualizations(
            model_class=config['model_class'],
            model_path=config['model_path'],
            dataset_folder=config['dataset_folder'],
            dataset_name=config['dataset_name']
        )


if __name__ == "__main__":
    run_visualization()