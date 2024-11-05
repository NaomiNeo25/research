import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class BLSTM2DCNN(BaseModel):
    def __init__(self, embedding_layer, num_classes, hidden_dim=128):
        super().__init__()
        self.embedding = embedding_layer
        self.gaussian_noise = GaussianNoise(0.2)
        self.dropout_embed = nn.Dropout(0.5)

        # BLSTM layer with proper number of layers and dropout
        self.blstm = nn.LSTM(
            input_size=300,  # embedding dimension
            hidden_size=hidden_dim,
            num_layers=2,  # Add multiple layers to properly use dropout
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # CNN layers with proper dimensionality
        self.conv1 = nn.Conv2d(1, 100, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(100, 100, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout_penultimate = nn.Dropout(0.2)

        # Adaptive pooling to handle variable sequence lengths
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Calculate final output size
        with torch.no_grad():
            # Create dummy input to calculate output size
            x = torch.zeros(1, 1, 64, hidden_dim * 2)
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.adaptive_pool(x)
            self.fc_input_size = x.numel()

        self.fc = nn.Linear(self.fc_input_size, num_classes)
        self._initialize_weights()

    def forward(self, x):
        # Get batch size and sequence length
        batch_size, seq_len = x.size()

        # Embedding and noise
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = self.gaussian_noise(x)
        x = self.dropout_embed(x)

        # Pack padded sequence for LSTM
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=[seq_len] * batch_size,
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
        cnn_in = lstm_out.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_dim * 2)

        # CNN layers
        conv1_out = F.relu(self.conv1(cnn_in))
        conv2_out = F.relu(self.conv2(conv1_out))
        pooled = self.pool(conv2_out)

        # Adaptive pooling to handle variable sequence lengths
        pooled = self.adaptive_pool(pooled)

        # Flatten and apply dropout
        flattened = pooled.view(batch_size, -1)
        dropped = self.dropout_penultimate(flattened)

        # Final classification
        out = self.fc(dropped)

        return out

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class BLSTM2DCNNWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(BLSTM2DCNNWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = nn.Linear(embedding_dim, 1)
        self.blstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv = nn.Conv2d(1, 128, kernel_size=(3, hidden_dim * 2))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.fc = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Embedding
        embeds = self.embedding(x)  # (batch_size, seq_length, embedding_dim)

        # Attention over embeddings
        attn_weights = F.softmax(self.attention(embeds), dim=1)  # (batch_size, seq_length, 1)
        embeds = embeds * attn_weights  # Element-wise multiplication

        # BLSTM
        blstm_out, _ = self.blstm(embeds)
        blstm_out = blstm_out.unsqueeze(1)  # (batch_size, 1, seq_length, hidden_dim * 2)

        # 2D CNN
        conv_out = F.relu(self.conv(blstm_out))
        pooled = self.pool(conv_out).squeeze(3).squeeze(2)

        # Classification
        output = self.dropout(pooled)
        output = self.fc(output)
        output = self.softmax(output)

        return output


class ParallelBLSTMCNNWithAttention(BaseModel):
    def __init__(self, embedding_layer, num_classes, hidden_dim=128):
        super().__init__()
        self.embedding = embedding_layer
        self.gaussian_noise = GaussianNoise(0.2)
        self.dropout_embed = nn.Dropout(0.5)

        # BLSTM branch
        self.blstm = nn.LSTM(
            input_size=300,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # CNN branch
        self.conv = nn.Conv2d(1, 128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2 + 128, 1)

        # Output layer
        self.dropout_penultimate = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim * 2 + 128, num_classes)
        self._initialize_weights()

    def forward(self, x):
        # Embedding and noise
        x = self.embedding(x)
        x = self.gaussian_noise(x)
        x = self.dropout_embed(x)

        # BLSTM branch
        lstm_out, _ = self.blstm(x)

        # CNN branch
        cnn_in = x.unsqueeze(1)
        conv_out = self.conv(cnn_in)
        conv_out = F.relu(conv_out)
        conv_out = self.pool(conv_out)

        # Combine features
        batch_size = x.size(0)
        conv_out = conv_out.view(batch_size, 128, -1).mean(dim=2)
        combined = torch.cat((lstm_out[:, -1, :], conv_out), dim=1)

        # Attention
        attention_weights = F.softmax(self.attention(combined), dim=1)
        attended = combined * attention_weights

        # Classification
        attended = self.dropout_penultimate(attended)
        out = self.fc(attended)

        return out

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs):
        attn_energies = self.attention(encoder_outputs).squeeze(-1)
        attn_weights = F.softmax(attn_energies, dim=1)
        attended = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return attended


class ModelArchitecture1(BaseModel):
    def __init__(self, embedding_layer, num_classes, hidden_dim=128):
        super().__init__()
        self.embedding = embedding_layer
        self.gaussian_noise = GaussianNoise(0.2)
        self.dropout_embed = nn.Dropout(0.5)

        # BLSTM
        self.blstm = nn.LSTM(
            input_size=300,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # CNN
        self.conv = nn.Conv2d(1, 128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        # Self-attention
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Output
        self.dropout_penultimate = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)
        self._initialize_weights()

    def forward(self, x):
        # Embedding and noise
        x = self.embedding(x)
        x = self.gaussian_noise(x)
        x = self.dropout_embed(x)

        # BLSTM
        lstm_out, _ = self.blstm(x)

        # Self-attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended = torch.bmm(attention_weights.transpose(1, 2), lstm_out)

        # CNN
        cnn_in = lstm_out.unsqueeze(1)
        conv_out = self.conv(cnn_in)
        conv_out = F.relu(conv_out)
        conv_out = self.pool(conv_out)

        # Combine features
        batch_size = x.size(0)
        conv_out = conv_out.view(batch_size, -1)
        combined = torch.cat((attended.squeeze(1), conv_out), dim=1)

        # Classification
        combined = self.dropout_penultimate(combined)
        out = self.fc(combined)

        return out


class ModelArchitecture2(BaseModel):
    def __init__(self, embedding_layer, num_classes, hidden_dim=128):
        super().__init__()
        self.embedding = embedding_layer
        self.gaussian_noise = GaussianNoise(0.2)
        self.dropout_embed = nn.Dropout(0.5)

        # Hierarchical attention
        self.word_attention = nn.Linear(300, hidden_dim)
        self.word_context = nn.Linear(hidden_dim, 1)

        # BLSTM
        self.blstm = nn.LSTM(
            input_size=300,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Output
        self.dropout_penultimate = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self._initialize_weights()

    def forward(self, x):
        # Embedding and noise
        x = self.embedding(x)
        x = self.gaussian_noise(x)
        x = self.dropout_embed(x)

        # Word-level attention
        word_weights = torch.tanh(self.word_attention(x))
        word_weights = self.word_context(word_weights)
        word_weights = F.softmax(word_weights, dim=1)
        word_vector = torch.bmm(word_weights.transpose(1, 2), x)

        # BLSTM
        lstm_out, _ = self.blstm(word_vector)

        # Classification
        final_state = lstm_out[:, -1, :]
        final_state = self.dropout_penultimate(final_state)
        out = self.fc(final_state)

        return out

class HierarchicalAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HierarchicalAttention, self).__init__()
        self.attention = nn.Linear(input_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        attn = torch.tanh(self.attention(x))  # (batch_size, seq_len, hidden_size)
        attn_weights = F.softmax(self.context_vector(attn).squeeze(-1), dim=1)  # (batch_size, seq_len)
        attended = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (batch_size, input_size)
        return attended

class BLSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super(BLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, embedded_text):
        lstm_out, _ = self.lstm(embedded_text)
        return lstm_out  # (batch_size, seq_len, hidden_size * 2)

class ClassificationLayer(nn.Module):
    def __init__(self, num_filters, num_classes):
        super(ClassificationLayer, self).__init__()
        # Ensure the input dim matches the number of filters after CNN + MaxPooling
        self.fc = nn.Linear(num_filters, num_classes)  # num_filters should match the output from CNN+Pooling

    def forward(self, pooled_out):
        return self.fc(pooled_out)

class CNNLayer(nn.Module):
    def __init__(self, input_dim, num_filters, filter_size):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv1d(input_dim, num_filters, filter_size, padding=1)

    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        # Transpose to (batch_size, hidden_size * 2, seq_len) for 1D convolution
        lstm_out = lstm_out.transpose(1, 2)
        conv_out = self.conv(lstm_out)
        return conv_out

class MaxPoolingLayer(nn.Module):
    def __init__(self, pool_size):
        super(MaxPoolingLayer, self).__init__()
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, conv_out):
        pooled_out = self.pool(conv_out)
        return pooled_out.squeeze(2)

class BLSTM_CNN_Attention_Model(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_filters, filter_size, pool_size, num_classes):
        super(BLSTM_CNN_Attention_Model, self).__init__()

        # BLSTM layer
        self.blstm = BLSTMEncoder(embedding_dim, hidden_size)

        # Self-attention layer after BLSTM
        self.self_attention = SelfAttention(hidden_size * 2)

        # CNN layer
        self.cnn = CNNLayer(hidden_size * 2, num_filters, filter_size)

        # Max pooling layer
        self.pool = MaxPoolingLayer(pool_size)

        # Classification layer
        self.classifier = ClassificationLayer(num_filters, num_classes)

    def forward(self, input_embeddings):
        # Step 1: Process through BLSTM
        lstm_out = self.blstm(input_embeddings)

        # Step 2: Apply self-attention after BLSTM
        context_vector = self.self_attention(lstm_out)

        # Step 3: Apply 2D convolution on the LSTM output
        conv_out = self.cnn(lstm_out)

        # Step 4: Max pooling
        pooled_out = self.pool(conv_out)

        # Step 5: Classification
        logits = self.classifier(pooled_out)
        return logits


class BLSTM_CNN_Model(BaseModel):
    def __init__(self, embedding_layer, num_classes, hidden_dim=128):
        super().__init__()
        self.embedding = embedding_layer
        self.gaussian_noise = GaussianNoise(0.2)
        self.dropout_embed = nn.Dropout(0.5)

        # BLSTM
        self.blstm = nn.LSTM(
            input_size=300,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # CNN
        self.conv = nn.Conv2d(1, 128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        # Output
        self.dropout_penultimate = nn.Dropout(0.2)
        cnn_output_size = self._get_conv_output_size(hidden_dim)
        self.fc = nn.Linear(cnn_output_size, num_classes)
        self._initialize_weights()

    def _get_conv_output_size(self, hidden_dim):
        x = torch.randn(1, 1, 64, hidden_dim * 2)
        x = self.conv(x)
        x = self.pool(x)
        return x.numel()

    def forward(self, x):
        # Embedding and noise
        x = self.embedding(x)
        x = self.gaussian_noise(x)
        x = self.dropout_embed(x)

        # BLSTM
        lstm_out, _ = self.blstm(x)

        # CNN
        cnn_in = lstm_out.unsqueeze(1)
        conv_out = self.conv(cnn_in)
        conv_out = F.relu(conv_out)
        conv_out = self.pool(conv_out)

        # Classification
        batch_size = x.size(0)
        conv_out = conv_out.view(batch_size, -1)
        conv_out = self.dropout_penultimate(conv_out)
        out = self.fc(conv_out)

        return out