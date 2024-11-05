import os
import torch
import re
from torch.utils.data import Dataset, DataLoader, random_split
import sentencepiece as spm
from collections import defaultdict
import numpy as np
import logging

# Load the BPE tokenizer
sp = spm.SentencePieceProcessor()
sp.Load('./bpe/bpe.model')
vocab_size = sp.piece_size()

# Define Dataset Classes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Base dataset class with common functionality."""

    def __init__(self, sp_model, max_seq_length=512):
        self.sp = sp_model
        self.max_seq_length = max_seq_length
        self.texts = []
        self.labels = []
        self.author_to_idx = {}
        self.num_classes = 0

    def _verify_labels(self):
        """Verify that labels are consecutive integers starting from 0."""
        unique_labels = sorted(set(self.labels))
        expected_labels = list(range(len(unique_labels)))
        if unique_labels != expected_labels:
            raise ValueError(f"Labels must be consecutive integers from 0 to {len(unique_labels) - 1}. "
                             f"Found labels: {unique_labels}")
        self.num_classes = len(unique_labels)
        logger.info(f"Verified {self.num_classes} classes with correct label indexing")

    def _process_text(self, text):
        """Process text into tokens with proper padding."""
        tokens = self.sp.encode(text)
        if self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        return torch.tensor(tokens, dtype=torch.long)

class GuardianDataset(BaseDataset):
    def __init__(self, root_dir, sp_model, max_seq_length=512):
        super().__init__(sp_model, max_seq_length)
        self.root_dir = root_dir

        # Initialize counters
        author_docs = defaultdict(int)

        # First pass: count documents per author
        for split in ['Books', 'Politics', 'Society', 'UK', 'World']:
            split_dir = os.path.join(self.root_dir, split)
            if not os.path.exists(split_dir):
                continue

            for author in os.listdir(split_dir):
                author_dir = os.path.join(split_dir, author)
                if not os.path.isdir(author_dir):
                    continue

                for file in os.listdir(author_dir):
                    if file.endswith('.txt'):
                        author_docs[author] += 1

        # Select top 50 authors
        top_authors = sorted(author_docs.items(), key=lambda x: x[1], reverse=True)[:50]
        self.author_to_idx = {author: idx for idx, (author, _) in enumerate(top_authors)}

        # Second pass: collect texts and labels
        for split in ['Books', 'Politics', 'Society', 'UK', 'World']:
            split_dir = os.path.join(self.root_dir, split)
            if not os.path.exists(split_dir):
                continue

            for author in os.listdir(split_dir):
                if author not in self.author_to_idx:
                    continue

                author_dir = os.path.join(split_dir, author)
                if not os.path.isdir(author_dir):
                    continue

                for file in os.listdir(author_dir):
                    if not file.endswith('.txt'):
                        continue

                    file_path = os.path.join(author_dir, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read().strip()
                        if text:  # Only add non-empty texts
                            self.texts.append(text)
                            self.labels.append(self.author_to_idx[author])
                    except Exception as e:
                        logger.warning(f"Error reading file {file_path}: {str(e)}")

        self._verify_labels()
        logger.info(f"Loaded {len(self.texts)} documents from {self.num_classes} authors")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self._process_text(self.texts[idx]), self.labels[idx]

class IMDb62Dataset(BaseDataset):
    def __init__(self, data_file, sp_model, max_seq_length=512):
        super().__init__(sp_model, max_seq_length)

        # Read and process data file
        author_reviews = defaultdict(list)
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    fields = line.strip().split('\t')
                    if len(fields) >= 6:
                        _, user_id, _, _, _, content = fields[:6]
                        if content.strip():  # Only add non-empty reviews
                            author_reviews[user_id].append(content)
        except Exception as e:
            raise RuntimeError(f"Error reading IMDB dataset: {str(e)}")

        # Select authors with sufficient reviews and create mapping
        authors_with_count = [(author, len(reviews)) for author, reviews in author_reviews.items()]
        top_authors = sorted(authors_with_count, key=lambda x: x[1], reverse=True)[:62]
        self.author_to_idx = {author: idx for idx, (author, _) in enumerate(top_authors)}

        # Collect texts and labels
        for author, reviews in author_reviews.items():
            if author in self.author_to_idx:
                self.texts.extend(reviews)
                self.labels.extend([self.author_to_idx[author]] * len(reviews))

        self._verify_labels()
        logger.info(f"Loaded {len(self.texts)} reviews from {self.num_classes} authors")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self._process_text(self.texts[idx]), self.labels[idx]

class Reuters50Dataset(BaseDataset):
    def __init__(self, root_dir, sp_model, split='train', max_seq_length=512):
        super().__init__(sp_model, max_seq_length)
        self.root_dir = os.path.join(root_dir, split)

        # Count documents per author
        author_docs = defaultdict(int)
        for author in os.listdir(self.root_dir):
            author_dir = os.path.join(self.root_dir, author)
            if not os.path.isdir(author_dir):
                continue
            for file in os.listdir(author_dir):
                if file.endswith('.txt'):
                    author_docs[author] += 1

        # Select top 50 authors
        top_authors = sorted(author_docs.items(), key=lambda x: x[1], reverse=True)[:50]
        self.author_to_idx = {author: idx for idx, (author, _) in enumerate(top_authors)}

        # Collect texts and labels
        for author in self.author_to_idx:
            author_dir = os.path.join(self.root_dir, author)
            if not os.path.isdir(author_dir):
                continue

            for file in os.listdir(author_dir):
                if not file.endswith('.txt'):
                    continue

                file_path = os.path.join(author_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    if text:  # Only add non-empty texts
                        self.texts.append(text)
                        self.labels.append(self.author_to_idx[author])
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {str(e)}")

        self._verify_labels()
        logger.info(f"Loaded {len(self.texts)} documents from {self.num_classes} authors")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self._process_text(self.texts[idx]), self.labels[idx]

class Blog50Dataset(BaseDataset):
    def __init__(self, root_dir, sp_model, max_seq_length=512):
        super().__init__(sp_model, max_seq_length)

        # Process blog posts
        logger.info(f"Initializing Blog50Dataset from {root_dir}")
        author_posts = self._collect_author_posts(root_dir)

        # Select and process top 50 authors
        self._process_authors(author_posts)

        # Verify labels
        self._verify_labels()
        logger.info(f"Loaded {len(self.texts)} posts from {self.num_classes} authors")
        logger.info(f"Label range: {min(self.labels)} to {max(self.labels)}")

    def _collect_author_posts(self, root_dir):
        """Collect posts for each author."""
        author_posts = defaultdict(list)
        processed_files = 0

        for file in os.listdir(root_dir):
            if not file.endswith('.xml'):
                continue

            try:
                author_id = file.split('.')[0]
                file_path = os.path.join(root_dir, file)

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Split content into posts
                posts = re.split(r'\nDATE:\s*[^\n]+\n', content)
                posts = [post.strip() for post in posts if post.strip()]

                if posts:  # Only add authors with non-empty posts
                    author_posts[author_id].extend(posts)
                    processed_files += 1

            except Exception as e:
                logger.warning(f"Error processing file {file}: {str(e)}")

        logger.info(f"Processed {processed_files} files, found {len(author_posts)} authors")
        return author_posts

    def _process_authors(self, author_posts):
        """Process authors and ensure consecutive label indexing."""
        # Get authors with post counts
        author_counts = [(author, len(posts)) for author, posts in author_posts.items()]
        logger.info(f"Author post counts: min={min(c for _, c in author_counts)}, "
                    f"max={max(c for _, c in author_counts)}, "
                    f"mean={sum(c for _, c in author_counts) / len(author_counts):.1f}")

        # Select top 50 authors
        top_authors = sorted(author_counts, key=lambda x: x[1], reverse=True)[:50]
        logger.info(f"Selected top 50 authors with post counts from "
                    f"{top_authors[-1][1]} to {top_authors[0][1]}")

        # Create consecutive label mapping
        self.author_to_idx = {author: idx for idx, (author, _) in enumerate(top_authors)}

        # Collect texts and labels
        self.texts = []
        self.labels = []
        for author, _ in top_authors:
            # Get up to 50 posts per author
            posts = author_posts[author][:50]
            self.texts.extend(posts)
            self.labels.extend([self.author_to_idx[author]] * len(posts))

        # Verify label consistency
        unique_labels = sorted(set(self.labels))
        expected_labels = list(range(len(unique_labels)))

        if unique_labels != expected_labels:
            logger.error("Label mismatch detected!")
            logger.error(f"Found labels: {unique_labels}")
            logger.error(f"Expected labels: {expected_labels}")
            raise ValueError(
                f"Labels are not consecutive integers. Found {len(unique_labels)} "
                f"unique labels from {min(unique_labels)} to {max(unique_labels)}, "
                f"expected {len(expected_labels)} labels from "
                f"{min(expected_labels)} to {max(expected_labels)}"
            )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return self._process_text(text), label

def _verify_dataset(dataset):
    """Utility function to verify dataset consistency."""
    # Check label range
    unique_labels = sorted(set(dataset.labels))
    expected_labels = list(range(len(unique_labels)))

    print(f"\nDataset Verification:")
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of unique labels: {len(unique_labels)}")
    print(f"Label range: {min(unique_labels)} to {max(unique_labels)}")

    # Check label distribution
    label_counts = Counter(dataset.labels)
    print("\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        print(f"Label {label}: {label_counts[label]} samples")

    # Verify consecutive labels
    if unique_labels != expected_labels:
        print("\nWarning: Non-consecutive labels detected!")
        print(f"Found labels: {unique_labels}")
        print(f"Expected labels: {expected_labels}")
        return False

    return True


# Example usage for testing the dataset
if __name__ == "__main__":
    import sys
    from collections import Counter

    if len(sys.argv) != 2:
        print("Usage: python blog_dataset.py <path_to_blog_data>")
        sys.exit(1)

    # Initialize tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load('bpe.model')

    # Create and verify dataset
    try:
        dataset = Blog50Dataset(sys.argv[1], sp)
        if _verify_dataset(dataset):
            print("\nDataset verification passed!")
        else:
            print("\nDataset verification failed!")
    except Exception as e:
        print(f"\nError creating dataset: {str(e)}")

def _verify_dataset(dataset):
    """Utility function to verify dataset consistency."""
    # Check label range
    unique_labels = sorted(set(dataset.labels))
    expected_labels = list(range(len(unique_labels)))

    print(f"\nDataset Verification:")
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of unique labels: {len(unique_labels)}")
    print(f"Label range: {min(unique_labels)} to {max(unique_labels)}")

    # Check label distribution
    label_counts = Counter(dataset.labels)
    print("\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        print(f"Label {label}: {label_counts[label]} samples")

    # Verify consecutive labels
    if unique_labels != expected_labels:
        print("\nWarning: Non-consecutive labels detected!")
        print(f"Found labels: {unique_labels}")
        print(f"Expected labels: {expected_labels}")
        return False

    return True


# Example usage for testing the dataset
if __name__ == "__main__":
    import sys
    from collections import Counter

    if len(sys.argv) != 2:
        print("Usage: python blog_dataset.py <path_to_blog_data>")
        sys.exit(1)

    # Initialize tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load('bpe.model')

    # Create and verify dataset
    try:
        dataset = Blog50Dataset(sys.argv[1], sp)
        if _verify_dataset(dataset):
            print("\nDataset verification passed!")
        else:
            print("\nDataset verification failed!")
    except Exception as e:
        print(f"\nError creating dataset: {str(e)}")

def get_dataloaders(dataset_class, dataset_args, batch_size=32, val_split=0.1):
    """Create train, validation, and test dataloaders with proper splits."""
    try:
        full_dataset = dataset_class(**dataset_args)

        # Compute split sizes
        total_samples = len(full_dataset)
        test_size = int(0.2 * total_samples)
        val_size = int(val_split * total_samples)
        train_size = total_samples - test_size - val_size

        # Create splits
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Create and return dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

        logger.info(f"Created dataloaders with {train_size}/{val_size}/{test_size} splits")
        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error(f"Error creating dataloaders: {str(e)}")
        raise
# Define collate function

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(t) for t in texts]
    max_length = max(lengths)

    padded_texts = torch.zeros(len(texts), max_length, dtype=torch.long)
    for i, text in enumerate(texts):
        end = lengths[i]
        padded_texts[i, :end] = text[:end]
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_texts, labels

