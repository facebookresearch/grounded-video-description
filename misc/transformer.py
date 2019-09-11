# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Originally from https://github.com/salesforce/densecap
"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# Last modified by Luowei Zhou on 12/27/2018

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import random
import string
import sys
import math
import uuid
import numpy as np

INF = 1e10

def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0, x.size(1))
        if x.is_cuda:
           positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())


    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(
                positions / 10000 ** ((channel - 1) / x.size(2)))
    return Variable(encodings)

def mask(targets, out):
    mask = (targets != 0)
    out_mask = mask.unsqueeze(-1).expand_as(out)
    return targets[mask], out[out_mask].view(-1, out.size(-1))

# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
# not exactly true but keep it for legacy reason
def matmul(x, y):
    if x.dim() == y.dim():
        return torch.matmul(x, y)
    if x.dim() == y.dim() - 1:
        return torch.matmul(x.unsqueeze(-2), y).squeeze(-2)
    return torch.matmul(x, y.unsqueeze(-2)).squeeze(-2)

class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio):
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x):
        return self.layernorm(x[0] + self.dropout(self.layer(*x)))

class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal):
        super(Attention, self).__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal

    def forward(self, query, key, value):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and (self is None or self.causal):
            tri = torch.ones(key.size(1), key.size(1)).triu(1) * INF
            if key.is_cuda:
                tri = tri.cuda(key.get_device())
            dot_products.data.sub_(tri.unsqueeze(0))
        return matmul(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)

class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False):
        super(MultiHead, self).__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal)
        self.wq = nn.Linear(d_key, d_key, bias=False)
        self.wk = nn.Linear(d_key, d_key, bias=False)
        self.wv = nn.Linear(d_value, d_value, bias=False)
        self.wo = nn.Linear(d_value, d_key, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        return self.wo(torch.cat([self.attention(q, k, v)
                          for q, k, v in zip(query, key, value)], -1))

class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super(EncoderLayer, self).__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x):
        return self.feedforward(self.selfattn(x, x, x))

class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super(DecoderLayer, self).__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio, causal=True),
            d_model, drop_ratio)
        self.attention = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x, encoding):
        x = self.selfattn(x, x, x)
        return self.feedforward(self.attention(x, encoding, encoding))

class Encoder(nn.Module):

    def __init__(self, d_model, d_hidden, n_vocab, n_layers, n_heads,
                 drop_ratio, pe):
        super(Encoder, self).__init__()
        # self.linear = nn.Linear(d_model*2, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for i in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)
        self.pe = pe

    def forward(self, x, mask=None):
        # x = self.linear(x)
        if self.pe:
            x = x+positional_encodings_like(x) # spatial configuration is already encoded
        # x = self.dropout(x) # dropout is already in the pool_embed layer
        if mask is not None:
            x = x*mask
        encoding = []
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x*mask
            encoding.append(x)
        return encoding

class Decoder(nn.Module):

    def __init__(self, d_model, d_hidden, vocab_size, n_layers, n_heads,
                 drop_ratio):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for i in range(n_layers)])
        self.out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(drop_ratio)
        self.d_model = d_model
        # self.vocab = vocab
        self.d_out = vocab_size

    def forward(self, x, encoding):
        x = F.embedding(x, self.out.weight * math.sqrt(self.d_model))
        x = x+positional_encodings_like(x)
        x = self.dropout(x)
        for layer, enc in zip(self.layers, encoding):
            x = layer(x, enc)
        return x

    def greedy(self, encoding, T):
        B, _, H = encoding[0].size()
        # change T to 20, max # of words in a sentence
        # T = 40
        # T *= 2
        prediction = Variable(encoding[0].data.new(B, T).long().fill_(
            0))
        hiddens = [Variable(encoding[0].data.new(B, T, H).zero_())
                   for l in range(len(self.layers) + 1)]
        embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        for t in range(T):
            if t == 0:
                hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(Variable(
                    encoding[0].data.new(B).long().fill_(
                        0)), embedW)
            else:
                hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(prediction[:, t - 1],
                                                                embedW)
            hiddens[0][:, t] = self.dropout(hiddens[0][:, t])
            for l in range(len(self.layers)):
                x = hiddens[l][:, :t + 1]
                x = self.layers[l].selfattn(hiddens[l][:, t], x, x)
                hiddens[l + 1][:, t] = self.layers[l].feedforward(
                    self.layers[l].attention(x, encoding[l], encoding[l]))

            _, prediction[:, t] = self.out(hiddens[-1][:, t]).max(-1)
        return prediction


class Transformer(nn.Module):

    def __init__(self, d_model, n_vocab_src, vocab_trg, d_hidden=2048,
                 n_layers=6, n_heads=8, drop_ratio=0.1, pe=False):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, d_hidden, n_vocab_src, n_layers,
                               n_heads, drop_ratio, pe)

    def forward(self, x):
        encoding = self.encoder(x)
        return encoding[-1]
        # return encoding[-1], encoding
        # return torch.cat(encoding, 2)

    def all_outputs(self, x):
        encoding = self.encoder(x)
        return encoding

class TransformerDecoder(nn.Module):

    def __init__(self, d_model, n_vocab_src, vocab_trg, d_hidden=2048,
                 n_layers=2, n_heads=6, drop_ratio=0.2):
        super(TransformerDecoder, self).__init__()
        self.decoder = Decoder(d_model, d_hidden, vocab_trg, n_layers,
                              n_heads, drop_ratio)
        self.n_layers = n_layers

    def forward(self, encoding, s, ss_ratio=1, infer=False, seq_length=20):
        if infer:
            greedy = self.decoder.greedy(encoding, seq_length)
            return greedy

        out = self.decoder(s[:, :-1].contiguous(), encoding)
        targets, out = mask(s[:, 1:].contiguous(), out)
        logits = self.decoder.out(out)
        assert ss_ratio == 1, 'scheduled sampling does not work under pytorch 0.4' # TODO, ss_ratio<1 triggered gradient issues
        return F.cross_entropy(logits, targets)
