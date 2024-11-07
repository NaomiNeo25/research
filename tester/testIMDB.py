
from models import *
from tester.test import ModelTester


def run_testing():
    # Initialize tester
    tester = ModelTester(
        model_dir="./experiment_results/models",
        results_dir="./test_results"
    )

    # Define datasets and models to test
    test_configs = [
        {
            'model_class': BLSTM_CNN,
            'model_path': './experiment_results/models/best_model_blogs_fold5_BLSTM_CNN.pth',
            'dataset_folder': './datasets/blogs',
            'dataset_name': 'blogs'
        },
        {
            'model_class': TextCNN,
            'model_path': './experiment_results/models/best_model_Guardian_fold5_TextCNN.pth',
            'dataset_folder': './datasets/Guardian/Guardian_original',
            'dataset_name': 'Guardian'
        },
        # Add more configurations as needed
    ]

    # Run tests
    results = []
    for config in test_configs:
        result = tester.test_model(
            model_class=config['model_class'],
            model_path=config['model_path'],
            dataset_folder=config['dataset_folder'],
            dataset_name=config['dataset_name']
        )
        if result:
            results.append(result)

    return results


if __name__ == "__main__":
    results = run_testing()

    # Print summary of results
    print("\nTesting Results Summary:")
    for result in results:
        print(f"\nModel: {result['model_name']}")
        print(f"Dataset: {result['dataset']}")
        print(f"F1 Score: {result['f1_score']:.4f}")