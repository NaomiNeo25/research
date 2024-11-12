
import torch
import torch.nn as nn
import torch.nn.functional as F



class BLSTM_TextCNN(nn.Module):
    def __init__(self, embedding_layer, num_classes, hidden_dim=128, kernel_sizes=[3, 4, 5], num_filters=100):
        super(BLSTM_TextCNN, self).__init__()
        self.embedding = embedding_layer
        embed_dim = embedding_layer.embedding_dim


        self.blstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )


        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_dim * 2, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])


        self.dropout = nn.Dropout(0.5)


        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x, lengths=None):

        x = self.embedding(x)


        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)


        lstm_out, _ = self.blstm(x)


        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)


        lstm_out = lstm_out.transpose(1, 2)


        conv_outs = [F.relu(conv(lstm_out)) for conv in self.convs]
        pooled_outs = [F.max_pool1d(out, kernel_size=out.size(2)).squeeze(2) for out in conv_outs]


        concat = torch.cat(pooled_outs, dim=1)


        dropped = self.dropout(concat)


        out = self.fc(dropped)

        return out

class ParallelCNNBLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, hidden_dim=128, kernel_sizes=[3, 4, 5], num_filters=100):
        super(ParallelCNNBLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)


        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])


        self.blstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)


        self.fc = nn.Linear(len(kernel_sizes) * num_filters + hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lengths=None):
        x = self.embedding(x)


        x_cnn = x.unsqueeze(1)
        cnn_outs = [F.relu(conv(x_cnn)).squeeze(3) for conv in self.convs]
        cnn_outs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in cnn_outs]
        cnn_out = torch.cat(cnn_outs, dim=1)

        blstm_out, _ = self.blstm(x)
        blstm_out = blstm_out[:, -1, :]


        combined = torch.cat((cnn_out, blstm_out), dim=1)
        combined = self.dropout(combined)


        out = self.fc(combined)
        return out

class ParallelCNNBLSTMWithAttention(nn.Module):
    def __init__(self, embedding_layer, num_classes, hidden_dim=128):
        super(ParallelCNNBLSTMWithAttention, self).__init__()
        self.embedding = embedding_layer
        embed_dim = embedding_layer.embedding_dim


        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, embed_dim), padding=(1, 0))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))


        self.blstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )


        combined_dim = 128 * 4 + hidden_dim * 2
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.Tanh(),
            nn.Linear(combined_dim, 1),
            nn.Softmax(dim=1)
        )


        self.fc = nn.Linear(combined_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lengths):
        batch_size = x.size(0)
        x_embedded = self.embedding(x)


        cnn_input = x_embedded.unsqueeze(1)
        cnn_input = cnn_input.permute(0, 1, 3, 2)
        cnn_out = F.relu(self.conv1(cnn_input))
        cnn_out = self.pool(cnn_out)
        cnn_out = self.adaptive_pool(cnn_out)
        cnn_out = cnn_out.view(batch_size, -1)


        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            x_embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (h_n, _) = self.blstm(packed_embedded)
        h_n = h_n.view(self.blstm.num_layers, 2, batch_size, self.blstm.hidden_size)
        h_n_last = h_n[-1]
        blstm_out = torch.cat((h_n_last[0], h_n_last[1]), dim=1)


        combined = torch.cat((cnn_out, blstm_out), dim=1)


        attention_weights = self.attention(combined)
        attended = combined * attention_weights


        dropped = self.dropout(attended)
        output = self.fc(dropped)

        return output

class ParallelCNNBLSTMWithPreConcatAttention(nn.Module):
    def __init__(self, embedding_layer, num_classes, hidden_dim=128, kernel_sizes=[3, 4, 5], num_filters=100):
        super(ParallelCNNBLSTMWithPreConcatAttention, self).__init__()
        self.embedding = embedding_layer
        embed_dim = embedding_layer.embedding_dim


        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.cnn_attention = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), num_filters * len(kernel_sizes)),
            nn.Tanh(),
            nn.Linear(num_filters * len(kernel_sizes), 1),
            nn.Softmax(dim=1)
        )


        self.blstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.blstm_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, 1),
            nn.Softmax(dim=1)
        )


        combined_dim = num_filters * len(kernel_sizes) + hidden_dim * 2
        self.fc = nn.Linear(combined_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lengths):
        batch_size = x.size(0)
        x_embedded = self.embedding(x)


        x_cnn = x_embedded.unsqueeze(1)
        cnn_outs = [F.relu(conv(x_cnn)).squeeze(3) for conv in self.convs]
        cnn_outs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in cnn_outs]
        cnn_out = torch.cat(cnn_outs, dim=1)


        cnn_attn_weights = self.cnn_attention(cnn_out)
        cnn_attended = cnn_out * cnn_attn_weights


        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            x_embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.blstm(packed_embedded)
        blstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)


        blstm_attn_weights = self.blstm_attention(blstm_out).squeeze(-1)
        blstm_attn_weights = blstm_attn_weights.masked_fill(
            torch.arange(blstm_out.size(1)).unsqueeze(0).expand(batch_size, -1).to(lengths.device) >= lengths.unsqueeze(1),
            float('-inf')
        )
        blstm_attn_weights = F.softmax(blstm_attn_weights, dim=1).unsqueeze(-1)
        blstm_attended = (blstm_out * blstm_attn_weights).sum(dim=1)


        combined = torch.cat((cnn_attended, blstm_attended), dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)

        return output

class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, embedding_layer, num_classes, word_hidden_dim=50, sent_hidden_dim=50, dropout=0.5,
                 sent_length=20, max_sent_length=25):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.embedding = embedding_layer
        embed_dim = embedding_layer.embedding_dim
        self.sent_length = sent_length
        self.max_sent_length = max_sent_length


        self.word_lstm = nn.LSTM(embed_dim, word_hidden_dim, bidirectional=True, batch_first=True)
        self.word_attention = nn.Sequential(
            nn.Linear(word_hidden_dim * 2, word_hidden_dim),
            nn.Tanh(),
            nn.Linear(word_hidden_dim, 1)
        )


        self.sent_lstm = nn.LSTM(word_hidden_dim * 2, sent_hidden_dim, bidirectional=True, batch_first=True)
        self.sent_attention = nn.Sequential(
            nn.Linear(sent_hidden_dim * 2, sent_hidden_dim),
            nn.Tanh(),
            nn.Linear(sent_hidden_dim, 1)
        )


        self.word_norm = nn.BatchNorm1d(word_hidden_dim * 2)
        self.sent_norm = nn.BatchNorm1d(sent_hidden_dim * 2)


        self.fc = nn.Linear(sent_hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        batch_size = x.size(0)
        seq_length = x.size(1)


        num_sentences = max(1, min(self.sent_length, seq_length // self.max_sent_length))
        words_per_sentence = max(1, min(self.max_sent_length, seq_length // num_sentences))


        min_seq_length = num_sentences * words_per_sentence
        if seq_length < min_seq_length:
            padding_length = min_seq_length - seq_length
            padding = torch.zeros(batch_size, padding_length, dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding], dim=1)


        x = x[:, :num_sentences * words_per_sentence]
        x = x.view(batch_size, num_sentences, words_per_sentence)


        x_reshaped = x.contiguous().view(batch_size * num_sentences, words_per_sentence)


        word_embeds = self.dropout(self.embedding(x_reshaped))


        word_lstm_out, _ = self.word_lstm(word_embeds)


        word_attn_weights = self.word_attention(word_lstm_out)
        word_attn_weights = F.softmax(word_attn_weights.squeeze(-1), dim=1)


        word_attn_out = torch.bmm(
            word_attn_weights.unsqueeze(1),
            word_lstm_out
        ).squeeze(1)


        word_attn_out = self.word_norm(word_attn_out)


        sent_inputs = word_attn_out.view(batch_size, num_sentences, -1)


        sent_lstm_out, _ = self.sent_lstm(sent_inputs)


        sent_attn_weights = self.sent_attention(sent_lstm_out)
        sent_attn_weights = F.softmax(sent_attn_weights.squeeze(-1), dim=1)


        doc_vector = torch.bmm(
            sent_attn_weights.unsqueeze(1),
            sent_lstm_out
        ).squeeze(1)


        doc_vector = self.sent_norm(doc_vector)


        doc_vector = self.dropout(doc_vector)
        output = self.fc(doc_vector)

        return output

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


