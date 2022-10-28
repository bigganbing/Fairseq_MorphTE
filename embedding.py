import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MorphTEmbedding(nn.Module):
    def __init__(self, num_embeddings, num_surfaces, embedding_dim, co_matrix, core_dim=None, order=3, rank=8, padding_idx=None,
                 max_norm=None,
                 norm_type=2., scale_grad_by_freq=False, sparse=False):
        super(MorphTEmbedding, self).__init__()

        self.num_embeddings = num_embeddings    # number of morphemes
        self.num_surfaces = num_surfaces        # number of words
        self.embedding_dim = embedding_dim      # word embedding dim
        self.padding_idx = padding_idx

        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.rank = rank
        self.order = order

        self.core_dim = math.ceil((embedding_dim) ** (1 / order))   # morpheme embedding dim
        if core_dim is not None:
            self.core_dim = core_dim

        if self.core_dim ** order > embedding_dim:
            print("Note that the resulting word embeddings will be truncated.")

        self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(rank, num_embeddings, self.core_dim))])   # morpheme embedding matrices

        self.register_buffer('co_matrix', co_matrix)    # Morpheme Index Matrix
        self.reset_parameters()

        self.ln = nn.LayerNorm(embedding_dim)


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight[0])
        with torch.no_grad():
            self.weight[0][:, 0].fill_(0)

        # nn.init.normal_(self.weight[0], mean=0, std=self.embedding_dim ** -0.5)
        # nn.init.constant_(self.weight[0][:, 0], 0)

    def get_ith_core_block(self, ith, core_inx):
        co_inx = core_inx[:, ith].unsqueeze(0).unsqueeze(-1).expand(self.rank, -1, self.core_dim)
        return self.weight[0].gather(1, co_inx)


    def forward_block(self, x):
        "just generate word embeddings for a batch"

        bs, seq_len = x.shape
        num_surfaces = bs * seq_len
        x = x.view(-1)
        core_inx = self.co_matrix.gather(0, x.unsqueeze(-1).expand(-1, 3))

        w = self.get_ith_core_block(0, core_inx)

        for i in range(1, self.order):
            w_ = self.get_ith_core_block(i, core_inx)
            w = w[:, :, :, None] * w_[:, :, None, :]
            w = w.view(self.rank, num_surfaces, -1)

        w = w.sum(0)
        w = w[:, :self.embedding_dim]
        w = w.view(bs, seq_len, -1)
        w = self.ln(w)

        return w


    def get_ith_core(self, ith):
        co_inx = self.co_matrix[:, ith].unsqueeze(0).unsqueeze(-1).expand(self.rank, -1, self.core_dim)
        return self.weight[0].gather(1, co_inx)


    def get_emb_matrix(self):
        "generate word embeddings for the entire vocabulary"

        w = self.get_ith_core(0)

        for i in range(1, self.order):
            w_ = self.get_ith_core(i)
            w = w[:, :, :, None] * w_[:, :, None, :]
            w = w.view(self.rank, self.num_surfaces, -1)

        w = w.sum(0)
        # w = w.mean(0)

        w = w[:, :self.embedding_dim]
        w = self.ln(w)

        return w


    def forward(self, x):

        w = self.get_emb_matrix()

        return F.embedding(
            x, w, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


class MorphLSTMEmbedding(nn.Module):
    def __init__(self, num_embeddings, num_surfaces, embedding_dim, co_matrix, order=3, padding_idx=None,
                 max_norm=None,
                 norm_type=2., scale_grad_by_freq=False, sparse=False):
        super(MorphLSTMEmbedding, self).__init__()

        self.num_surfaces = num_surfaces        # number of words
        self.num_embeddings = num_embeddings    # number of morphemes
        self.embedding_dim = embedding_dim      # word embedding dim
        self.padding_idx = padding_idx

        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.order = order
        self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))])
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim)

        self.register_buffer('co_matrix', co_matrix)
        self.reset_parameters()

        # self.ln = nn.LayerNorm(embedding_dim)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight[0])
        nn.init.constant_(self.weight[0][:, :3], 0)

    def get_ith_core(self, ith):
        co_inx = self.co_matrix[:, ith].unsqueeze(-1).expand(-1, self.embedding_dim)
        return self.weight[0].gather(0, co_inx)

    def get_emb_matrix(self):

        w = [self.get_ith_core(i) for i in range(self.order)]

        w = torch.stack(w, dim=0)

        o, w = self.rnn(w)

        w = w[0].squeeze(0)

        # w = self.ln(w)

        return w

    def forward(self, x):

        w = self.get_emb_matrix()

        return F.embedding(
            x, w, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


class NNEmbedding(nn.Module):
    # similar to nn.Embedding

    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(NNEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)
        self.sparse = sparse

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.weight[self.padding_idx], 0)

        # nn.init.normal_(self.weight)
        # if self.padding_idx is not None:
        #     with torch.no_grad():
        #         self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def get_emb_matrix(self):
        return self.weight

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)