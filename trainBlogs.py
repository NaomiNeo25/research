from models import *
from trainer import Trainer


def run_training():
    trainer = Trainer(
        save_dir="./experiment_results",
        batch_size=64,
        num_epochs=50,
        embed_dim=300,
        hidden_dim=128
    )

    # Define dataset folders
    # dataset_folders = [
    #     './datasets/blogs',
    #     './datasets/CCAT50',
    #     './datasets/IMDB62',
    #     './datasets/Guardian'
    # ]
    dataset_folders = [
        './datasets/blogs',
    ]

    model_classes = [
        TextCNN,
        BLSTM_TextCNN,
        ImprovedTextCNN,
        ParallelCNNBLSTM,
        ParallelCNNBLSTMWithAttention,
        ParallelCNNBLSTMWithPreConcatAttention,
        ParallelCNNBLSTMWithMHA_BeforeConcat,
        ParallelCNNBLSTMWithMHA_AfterConcat,
        HierarchicalAttentionNetwork,
        RCNN,
        BLSTM2DCNNWithMultiHeadAttention,
        ParallelBLSTMCNNWithAttention,
        HAN,
        BLSTM_CNN
    ]

    results = trainer.run_experiment(model_classes, dataset_folders)
    return results


if __name__ == "__main__":
    run_training()
