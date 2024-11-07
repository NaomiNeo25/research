import torch
from torch.utils.data import Dataset, DataLoader

# Define the TextDataset and collate_fn
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])  # Ensure text is a string
        label = self.labels[idx]

        # Tokenize text using BPE tokenizer
        tokens = self.tokenizer.encode(text)
        length = len(tokens)
        if length > self.max_length:
            tokens = tokens[:self.max_length]
            length = self.max_length

        # Pad tokens
        padding_length = self.max_length - length
        tokens = tokens + [0] * padding_length  # Assuming 0 is the padding index

        tokens = torch.tensor(tokens, dtype=torch.long)
        length = torch.tensor(length, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return tokens, label, length

def collate_fn(batch):
    tokens, labels, lengths = zip(*batch)
    tokens = torch.stack(tokens)
    labels = torch.stack(labels)
    lengths = torch.stack(lengths)
    return tokens, labels, lengths

