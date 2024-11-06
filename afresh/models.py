import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, embedding_layer=None, num_classes=None, vocab_size=None, embed_dim=None, kernel_sizes=[3, 4, 5], num_filters=100):
        super(TextCNN, self).__init__()
        if embedding_layer is not None:
            self.embedding = embedding_layer
            embed_dim = embedding_layer.embedding_dim
        else:
            if vocab_size is None or embed_dim is None:
                raise ValueError("vocab_size and embed_dim must be provided if embedding_layer is None")
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        if num_classes is None:
            raise ValueError("num_classes must be provided")
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x, lengths=None):
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embed_dim)
        conv_outs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, num_filters, seq_len-k+1), ...]
        pooled_outs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]  # [(batch_size, num_filters), ...]
        concat = torch.cat(pooled_outs, dim=1)  # (batch_size, num_filters * len(kernel_sizes))
        dropped = self.dropout(concat)
        out = self.fc(dropped)
        return out

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer."""

    def __init__(self, sigma=0.2):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x

class BaseModel(nn.Module):
    """Base class for all models with common initialization."""

    def __init__(self):
        super().__init__()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

class RCNN(BaseModel):
    def __init__(self, embedding_layer, num_classes, hidden_dim=128):
        super().__init__()
        self.embedding = embedding_layer
        self.lstm = nn.LSTM(
            embedding_layer.embedding_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        # Concatenated dimension will be embedding_dim + 2*hidden_dim
        total_dim = embedding_layer.embedding_dim + 2 * hidden_dim

        # Conv layer
        self.conv = nn.Conv1d(total_dim, hidden_dim, kernel_size=3, padding=1)

        # Add adaptive pooling for fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Final classification layer
        self.fc = nn.Linear(hidden_dim * 8, num_classes)

        self._initialize_weights()

    def forward(self, x, lengths):
        batch_size = x.size(0)

        # Embedding
        x_embed = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        # Pack padded sequence for LSTM
        packed_embed = nn.utils.rnn.pack_padded_sequence(
            x_embed,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM
        packed_output, _ = self.lstm(packed_embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )  # (batch_size, seq_len, 2*hidden_dim)

        # Concatenate embedding and LSTM outputs
        combined = torch.cat((x_embed, lstm_out), dim=2)  # (batch_size, seq_len, embed_dim + 2*hidden_dim)

        # Prepare for conv1d by changing dimension order
        combined = combined.permute(0, 2, 1)  # (batch_size, embed_dim + 2*hidden_dim, seq_len)

        # Apply convolution
        conv_out = F.relu(self.conv(combined))  # (batch_size, hidden_dim, seq_len)

        # Apply adaptive pooling for fixed output size
        pooled = self.adaptive_pool(conv_out)  # (batch_size, hidden_dim, 8)

        # Flatten and apply dropout
        flattened = pooled.view(batch_size, -1)  # (batch_size, hidden_dim * 8)
        dropped = self.dropout(flattened)

        # Classification
        output = self.fc(dropped)

        return output

class BLSTM2DCNN(BaseModel):
    def __init__(self, embedding_layer, num_classes, hidden_dim=128):
        super().__init__()
        self.embedding = embedding_layer
        self.dropout_embed = nn.Dropout(0.5)

        # BLSTM layer
        self.blstm = nn.LSTM(
            input_size=embedding_layer.embedding_dim,  # embedding dimension
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # CNN layers
        self.conv1 = nn.Conv2d(1, 100, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(100, 100, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout_penultimate = nn.Dropout(0.2)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Calculate the size of the fully connected layer input
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, 50, hidden_dim * 2)
            sample_output = self.conv_layers(sample_input)
            self.fc_input_size = sample_output.view(-1).size(0)

        self.fc = nn.Linear(self.fc_input_size, num_classes)
        self._initialize_weights()

    def conv_layers(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        return x

    def forward(self, x, lengths):
        # x: (batch_size, seq_len)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = self.dropout_embed(x)

        # Pack sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # BLSTM
        packed_lstm_out, _ = self.blstm(packed_x)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out,
            batch_first=True
        )  # (batch_size, seq_len, hidden_dim * 2)

        # Reshape for CNN
        lstm_out = lstm_out.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_dim * 2)

        # CNN layers
        conv_out = self.conv_layers(lstm_out)

        # Flatten and dropout
        flattened = conv_out.view(batch_size, -1)
        dropped = self.dropout_penultimate(flattened)

        # Fully connected layer
        out = self.fc(dropped)

        return out

class BLSTM2DCNNWithAttention(BaseModel):
    def __init__(self, embedding_layer, num_classes, hidden_dim=128):
        super().__init__()
        self.embedding = embedding_layer
        self.gaussian_noise = GaussianNoise(0.2)
        self.dropout_embed = nn.Dropout(0.5)

        # BLSTM layer
        self.blstm = nn.LSTM(
            input_size=embedding_layer.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # CNN layers
        self.conv = nn.Conv2d(1, 128, kernel_size=(3, hidden_dim * 2), padding=(1, 0))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout_penultimate = nn.Dropout(0.2)

        # Add adaptive pooling to ensure fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)  # Fixed output size

        # Calculate fc_input_size based on fixed output dimensions
        self.fc_input_size = 128 * 8  # 128 channels * 8 (adaptive pooling output size)

        # Output layer
        self.fc = nn.Linear(self.fc_input_size, num_classes)
        self._initialize_weights()

    def forward(self, x, lengths):
        batch_size = x.size(0)

        # Embedding and noise
        x_embedded = self.embedding(x)
        x_noisy = self.gaussian_noise(x_embedded)
        x_embedded = self.dropout_embed(x_noisy)

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            x_embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # BLSTM
        packed_output, _ = self.blstm(packed_embedded)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )  # (batch_size, seq_len, hidden_dim * 2)

        # Attention mechanism
        attention_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1).unsqueeze(2)  # (batch_size, seq_len, 1)
        attended = lstm_out * attention_weights  # (batch_size, seq_len, hidden_dim * 2)

        # Prepare for CNN
        cnn_in = attended.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_dim * 2)

        # CNN and pooling
        conv_out = F.relu(self.conv(cnn_in))  # (batch_size, 128, seq_len, 1)
        conv_out = self.pool(conv_out)  # (batch_size, 128, seq_len/2, 1)
        conv_out = conv_out.squeeze(-1)  # (batch_size, 128, seq_len/2)

        # Adaptive pooling to ensure fixed output size
        conv_out = self.adaptive_pool(conv_out)  # (batch_size, 128, 8)

        # Flatten and apply dropout
        flattened = conv_out.view(batch_size, -1)  # (batch_size, 128 * 8)
        dropped = self.dropout_penultimate(flattened)

        # Classification
        output = self.fc(dropped)

        return output

class ParallelBLSTMCNNWithAttention(BaseModel):
    def __init__(self, embedding_layer, num_classes, hidden_dim=128):
        super().__init__()
        self.embedding = embedding_layer
        self.gaussian_noise = GaussianNoise(0.2)
        self.dropout_embed = nn.Dropout(0.5)

        # BLSTM branch
        self.blstm = nn.LSTM(
            input_size=embedding_layer.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # CNN branch
        self.conv = nn.Conv2d(1, 128, kernel_size=(3, embedding_layer.embedding_dim), padding=(1, 0))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))  # Output size: (channels, 4)

        # Compute the output dimension after CNN branch for combining features
        conv_out_dim = 128 * 4  # 128 channels * 4 (from adaptive pooling)

        # Attention mechanism over combined features
        combined_dim = hidden_dim * 2 + conv_out_dim
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.Tanh(),
            nn.Linear(combined_dim, combined_dim),
            nn.Softmax(dim=1)
        )

        # Output layer
        self.dropout_penultimate = nn.Dropout(0.2)
        self.fc = nn.Linear(combined_dim, num_classes)
        self._initialize_weights()

    def conv_layers(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)  # Shape: (batch_size, 128, 1, 4)
        x = x.squeeze(2)  # Remove the height dimension, shape: (batch_size, 128, 4)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 128 * 4)
        return x

    def forward(self, x, lengths):
        batch_size = x.size(0)

        # Embedding and noise
        x_embedded = self.embedding(x)
        x_noisy = self.gaussian_noise(x_embedded)
        x_embedded = self.dropout_embed(x_noisy)

        # BLSTM branch
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            x_embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (h_n, _) = self.blstm(packed_embedded)
        # Get the last hidden state from both directions
        h_n = h_n.view(self.blstm.num_layers, 2, batch_size, self.blstm.hidden_size)
        h_n_last = h_n[-1]  # Last layer
        lstm_out = torch.cat((h_n_last[0], h_n_last[1]), dim=1)  # Shape: (batch_size, hidden_dim * 2)

        # CNN branch
        cnn_in = x_embedded.unsqueeze(1)  # Shape: (batch_size, 1, seq_len, embed_dim)
        cnn_in = cnn_in.permute(0, 1, 3, 2)  # Shape: (batch_size, 1, embed_dim, seq_len)
        conv_out = self.conv_layers(cnn_in)  # Shape: (batch_size, conv_out_dim)

        # Combine features
        combined = torch.cat((lstm_out, conv_out), dim=1)  # Shape: (batch_size, combined_dim)

        # Attention over combined features
        attention_weights = self.attention(combined)  # Shape: (batch_size, combined_dim)
        attended = combined * attention_weights  # Element-wise multiplication

        # Classification
        attended = self.dropout_penultimate(attended)
        output = self.fc(attended)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value, mask=None):
        attn_output, attn_weights = self.multihead_attn(query, key, value, attn_mask=mask)
        return attn_output, attn_weights

class BLSTM2DCNNWithMultiHeadAttention(BaseModel):
    def __init__(self, embedding_layer, num_classes, hidden_dim=128, num_heads=4):
        super().__init__()
        self.embedding = embedding_layer
        self.gaussian_noise = GaussianNoise(0.2)
        self.dropout_embed = nn.Dropout(0.5)

        # BLSTM layer
        self.blstm = nn.LSTM(
            input_size=embedding_layer.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Multi-Head Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # CNN layers
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, hidden_dim * 2), padding=(1, 0))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout_penultimate = nn.Dropout(0.2)

        # Add adaptive pooling to ensure fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)  # Fixed output size

        # Calculate fc_input_size based on fixed output dimensions
        self.fc_input_size = 128 * 8  # 128 channels * 8 (adaptive pooling output size)

        # Output layer
        self.fc = nn.Linear(self.fc_input_size, num_classes)
        self._initialize_weights()

    def forward(self, x, lengths):
        batch_size = x.size(0)

        # Embedding and noise
        x_embedded = self.embedding(x)
        x_noisy = self.gaussian_noise(x_embedded)
        x_embedded = self.dropout_embed(x_noisy)

        # Pack sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x_embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # BLSTM
        packed_output, _ = self.blstm(packed_x)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )  # (batch_size, seq_len, hidden_dim * 2)

        # Create attention mask for padding
        mask = torch.arange(lstm_out.size(1), device=lstm_out.device)[None, :] >= lengths[:, None]

        # Multi-Head Attention
        attn_output, _ = self.multihead_attn(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=mask
        )  # (batch_size, seq_len, hidden_dim * 2)

        # Prepare for CNN
        cnn_in = attn_output.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_dim * 2)

        # CNN layers
        conv_out = F.relu(self.conv1(cnn_in))  # (batch_size, 128, seq_len, 1)
        conv_out = self.pool(conv_out)  # (batch_size, 128, seq_len/2, 1)
        conv_out = conv_out.squeeze(-1)  # (batch_size, 128, seq_len/2)

        # Adaptive pooling to ensure fixed output size
        conv_out = self.adaptive_pool(conv_out)  # (batch_size, 128, 8)

        # Flatten and dropout
        flattened = conv_out.view(batch_size, -1)  # (batch_size, 128 * 8)
        dropped = self.dropout_penultimate(flattened)

        # Classification
        output = self.fc(dropped)

        return output

class HAN(nn.Module):
    def __init__(self, embedding_layer=None, num_classes=None, vocab_size=None, embed_dim=None, hidden_dim=128, dropout=0.5):
        super(HAN, self).__init__()

        if embedding_layer is not None:
            self.embedding = embedding_layer
            embed_dim = embedding_layer.embedding_dim
        else:
            if vocab_size is None or embed_dim is None:
                raise ValueError("vocab_size and embed_dim must be provided if embedding_layer is None")
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Word Encoder
        self.word_lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.word_attention = nn.Linear(hidden_dim * 2, 1)

        # Sentence Encoder
        self.sentence_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.sentence_attention = nn.Linear(hidden_dim * 2, 1)

        # Classifier
        if num_classes is None:
            raise ValueError("num_classes must be provided")
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        # x: (batch_size, seq_len)
        batch_size = x.size(0)

        # Embedding and dropout
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        # Word Encoder
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.word_lstm(packed_embedded)
        word_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # (batch_size, seq_len, hidden_dim*2)

        # Word Attention
        word_attn_scores = self.word_attention(word_out).squeeze(-1)  # (batch_size, seq_len)
        word_attn_weights = F.softmax(word_attn_scores, dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        word_context = word_out * word_attn_weights  # (batch_size, seq_len, hidden_dim*2)
        word_representation = word_context.sum(dim=1)  # (batch_size, hidden_dim*2)

        # Sentence Encoder (Assuming single sentence for simplicity)
        sentence_input = word_representation.unsqueeze(1)  # (batch_size, 1, hidden_dim*2)
        sentence_out, _ = self.sentence_lstm(sentence_input)  # (batch_size, 1, hidden_dim*2)

        # Sentence Attention
        sentence_attn_scores = self.sentence_attention(sentence_out).squeeze(-1)  # (batch_size, 1)
        sentence_attn_weights = F.softmax(sentence_attn_scores, dim=1).unsqueeze(-1)  # (batch_size, 1, 1)
        sentence_context = sentence_out * sentence_attn_weights  # (batch_size, 1, hidden_dim*2)
        sentence_representation = sentence_context.sum(dim=1)  # (batch_size, hidden_dim*2)

        # Classification
        output = self.fc(sentence_representation)  # (batch_size, num_classes)
        return output