import torch
import onmt

import torch.nn as nn
from torch.nn.functional import relu, max_pool2d
import numpy as np
from onmt.utils.misc import sequence_mask
from onmt.modules.embeddings import PositionalEncoding
from onmt.encoders.transformer import TransformerEncoder


class GatedPositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        super(GatedPositionalEncoding, self).__init__()
        self.pos_encoding = PositionalEncoding(dropout, dim, max_len)
        self.linear = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
        """
        gate = self.sigmoid(self.linear(emb))
        enc_emb = self.pos_encoding(emb.permute(1, 0, 2)).permute(1, 0, 2)
        return (1 - gate) * emb + gate * enc_emb

    def update_dropout(self, dropout):
        self.position_encoding.dropout.p = dropout


class HighwayLayer(nn.Module):
    def __init__(self, n_filters, dropout):
        super(HighwayLayer, self).__init__()
        self.n_filters = n_filters
        self.layer = nn.Linear(self.n_filters, 2 * self.n_filters)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        g, y = self.layer(x).split(self.n_filters, 2)
        g = self.sigmoid(g)
        out = g * self.relu(y) + (1 - g) * x
        out = self.dropout(out)
        return out

    def update_dropout(self, dropout):
        self.dropout.p = dropout


class ConvNet(nn.Module):
    def __init__(self, embedding_dim, filters_width, dropout):
        super(ConvNet, self).__init__()
        self.filters_width = filters_width
        self.embedding_dim = embedding_dim

        self.convolutions = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        for i in range(len(self.filters_width)):
            padding = nn.ZeroPad2d((0, 0, int((i + 1) / 2),
                                    i - int((i + 1) / 2),
                                    ))
            convolution = nn.Conv2d(1, self.filters_width[i], (i + 1, self.embedding_dim),
                                    padding=(0, 0))

            self.convolutions.append(nn.Sequential(padding, convolution))

        # self.__init_weights()

    # def __init_weights(self, init_range=0.1):
    #
    #     for i in range(len(self.filters_width)):
    #         getattr(self, "conv_layer{}".format(i)).weight.data.uniform_(-init_range, init_range)

    def forward(self, x):
        """
        Single-layer convolution over kernel with width ... + ReLU
        ----------
        @params
        x: tensor, with dimension (batch_size ,seq_len, embedding_size)
        @return
        out: tensor, with dimension (batch_size, seq_len, N), where
        N = \sum self.n_filters as feature
        ----------
        """
        # x: tensor, with dimension (batch_size, input_channels ,seq_len, embedding_size)
        # batch_size = x.size(0)
        # seq_len = x.size(2)
        convolved_seq = []
        for i in range(len(self.filters_width)):
            y = self.convolutions[i](x)
            convolved_seq.append(y)

        convolved_seq = torch.cat(convolved_seq, dim=1)
        convolved_seq = convolved_seq.permute(0, 3, 2, 1)
        convolved_seq = self.relu(convolved_seq)
        convolved_seq = self.dropout(convolved_seq)

        return convolved_seq

    def update_dropout(self, dropout):
        self.dropout.p = dropout


class ConvTransformerEncoder(TransformerEncoder):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions,
                 filters_width=(200, 200, 250, 250, 300, 300, 300, 300),
                 hw_layers=4, conv_pooling=5):
        super().__init__(num_layers, d_model, heads, d_ff, dropout,
                         attention_dropout, embeddings, max_relative_positions)

        self.conv_pooling = conv_pooling
        self.max_pooling = nn.MaxPool2d((self.conv_pooling, 1), stride=(self.conv_pooling, 1), ceil_mode=True) \
            if self.conv_pooling > 1 else None

        self.conv_net = ConvNet(embeddings.embedding_size, filters_width, dropout) \
            if int(np.sum(filters_width)) > 0 else None

        self.hw_dim = int(np.sum(filters_width)) if int(np.sum(filters_width)) > 0 else embeddings.embedding_size

        self.highway_net = nn.Sequential(*[HighwayLayer(self.hw_dim, dropout)
                                           for i in range(hw_layers)]) if hw_layers > 0 else None

        self.linear_mapping = nn.Linear(self.hw_dim, d_model) if self.hw_dim != d_model else None

        self.pos_encoding = PositionalEncoding(dropout, d_model) if not embeddings.position_encoding else None
        # if self.pos_encoding:
        #     print("Convolutional encoding works")
        # else:
        #     print("NOOO Convolutional")

        # self.gated_pos_encoding = GatedPositionalEncoding(dropout, d_model) if gated_enc else None

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        # print("First:", lengths, lengths.dtype)
        emb = self.embeddings(src)
        # print("before", emb.size())

        if self.conv_net or self.max_pooling or self.highway_net or self.linear_mapping:
            emb = emb.permute(1, 0, 2)

        if self.conv_net or self.max_pooling:
            emb = emb.unsqueeze(1)

        if self.conv_net:
            emb = self.conv_net(emb)

        if self.max_pooling:
            emb = self.max_pooling(emb)

        if self.conv_net or self.max_pooling:
            emb = emb.squeeze(1)

        if self.highway_net:
            emb = self.highway_net(emb)

        if self.linear_mapping:
            emb = self.linear_mapping(emb)

        if self.conv_net or self.max_pooling or self.highway_net or self.linear_mapping:
            emb = emb.permute(1, 0, 2)
        # print(emb.size())
        if self.pos_encoding:
            emb = self.pos_encoding(emb)

        out = emb.transpose(0, 1).contiguous()
        # print("-----", out.size())
        # out = emb.contiguous()
        len_type = lengths.dtype
        if self.max_pooling:
            lengths = torch.ceil(lengths.float() / self.conv_pooling).to(dtype=len_type)

        if lengths.max().tolist() != out.size(1):
                    # print("ERRoor", lengths.max().tolist(), out.size(1))
            lengths = torch.tensor(emb.size(1) * [emb.size(0)], device=emb.device)
        mask = ~sequence_mask(lengths).unsqueeze(1)
        # print("Second:", lengths, lengths.dtype, mask.size(), out.size())
        # print("")
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        # print(emb.size(), out.transpose(0, 1).contiguous().size(), lengths.size())

        return emb, out.transpose(0, 1).contiguous(), lengths

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
                                        is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            tuple(map(lambda x: int(x * opt.n_conv), (200, 200, 250, 250, 300, 300, 300, 300))),
            opt.hw_layers,
            opt.conv_pooling
        )

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        self.pos_encoding.update_dropout(dropout)
        self.conv_net.update_dropout(dropout) if self.conv_net else None
        if self.highway_net:
            for layer in self.highway_net:
                layer.update_dropout(dropout)

        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
