import os
import pandas as pd
import sentencepiece as spm


def train_bpe_tokenizer(dataset_folders, model_prefix='bpe', vocab_size=32000):
    # Check if the model already exists
    model_file = f"{model_prefix}.model"
    if os.path.exists(model_file):
        print(f"Tokenizer model '{model_file}' already exists. Skipping training.")
        return  # Skip training if model already exists

    # Combine all text data into a single file for training
    combined_text_file = 'combined_texts.txt'
    with open(combined_text_file, 'w', encoding='utf-8') as outfile:
        for dataset_folder in dataset_folders:
            for split in ['train.csv', 'validation.csv', 'test.csv']:
                csv_file = os.path.join(dataset_folder, split)
                if os.path.exists(csv_file):
                    data = pd.read_csv(csv_file)
                    texts = data['text'].tolist()
                    for text in texts:
                        outfile.write(str(text) + '\n')
                else:
                    print(f'File {csv_file} does not exist.')

    # Train the BPE tokenizer using the combined text file
    spm.SentencePieceTrainer.Train(
        input=combined_text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe'
    )
    print(f"Tokenizer model '{model_file}' trained and saved.")

def load_bpe_tokenizer(model_file='./../bpe/bpe.model'):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    return sp

if __name__ == "__main__":

    dataset_folders = [
        './../datasets/blogs',
        './../datasets/CCAT50',
        '.../datasets/IMDB62',
        './../datasets/Guardian'
    ]
    train_bpe_tokenizer(dataset_folders)
