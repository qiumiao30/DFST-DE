# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 20:47
# @File    : convformer.py
import torch
from torch import nn
import torch.nn.functional as F

from .modules import FeatureAttentionLayer

import numpy as np
from torch.autograd import Variable
from datetime import datetime

import math
import random
# import tqdm


class Action(nn.Module):
    def __init__(self, n_features, n_segment=8, shift_div=8):
        super(Action, self).__init__()
        self.n_segment = n_segment
        self.in_channels = n_features
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.reduced_channels = self.in_channels // 2
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fold = self.in_channels // shift_div

        # shifting
        self.action_shift = nn.Conv1d(
            self.in_channels, self.in_channels,
            kernel_size=self.kernel_size, padding=1, groups=self.in_channels,
            bias=False)
        self.action_shift.weight.requires_grad = True
        self.action_shift.weight.data.zero_()
        self.action_shift.weight.data[:self.fold, 0, 2] = 1  # shift left
        self.action_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1  # shift right

        if 2 * self.fold < self.in_channels:
            self.action_shift.weight.data[2 * self.fold:, 0, 1] = 1  # fixed

        # motion excitation
        self.pad = (0, 1)
        self.action_p3_squeeze = nn.Conv1d(self.in_channels, self.reduced_channels, kernel_size=1, stride=1,
                                           bias=False, padding=0)
        self.action_p3_bn1 = nn.BatchNorm1d(self.reduced_channels)
        self.action_p3_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=self.kernel_size,
                                         stride=1, bias=False, padding=1, groups=self.reduced_channels)
        self.action_p3_expand = nn.Conv1d(self.reduced_channels, self.in_channels, kernel_size=1, stride=1,
                                          bias=False, padding=0)
        print('=> Using ACTION')

    def forward(self, x):
        x_shift = x.permute(0, 2, 1)

        # # 1D convolution: motion excitation
        x3 = self.action_p3_squeeze(x_shift)
        x3 = self.action_p3_bn1(x3)
        n, c, t = x3.size()
        x3_plus0, x_left = x3.split([t - 1, 1], dim=2)
        x3_plus1 = self.action_p3_conv1(x3)

        _, x3_plus1 = x3_plus1.split([1, t - 1], dim=2)
        x_p3 = x3_plus1 - x3_plus0
        # x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)
        x_p3 = torch.cat([x_p3, x_left], 2)
        # x_p3 = self.avg_pool(x_p3.view(n, c, t))
        x_p3 = self.action_p3_expand(x_p3)
        x_p3 = self.sigmoid(x_p3)
        x_p3 = x_shift * x_p3

        # out1 = x_p2
        out = x_p3.permute(0, 2, 1)

        return out

class SEModule(nn.Module):
    def __init__(self, channels, reduction=3):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x + x

# Self Attention Class
class SelfAttentionConv(nn.Module):
    def __init__(self, k, headers, seq_length, kernel_size=5, mask_next=True, mask_diag=False):
        super().__init__()

        self.k, self.headers, self.kernel_size = k, headers, kernel_size
        self.mask_next = mask_next
        self.mask_diag = mask_diag

        h = headers

        # Query, Key and Value Transformations

        padding = (kernel_size - 1) * 2
        self.padding_opertor = nn.ConstantPad1d((padding, 0), 0)  # (padding, 0)--->(左填充，右填充)

        self.toqueries = nn.Conv1d(k, k * h, kernel_size, padding=0, bias=True, dilation=2)  # (k=in_channel, k*h=out_channel)
        self.tokeys = nn.Conv1d(k, k * h, kernel_size, padding=0, bias=True, dilation=2)
        self.tovalues = nn.Conv1d(k, k * h, kernel_size=1, padding=0, bias=False)  # No convolution operated

        self.linear = nn.Conv1d(k, k*h, kernel_size=1, padding=0, bias=False)


        #### initial
        # self.abs_pos_emb_key = nn.Embedding(seq_length, k * h)
        # self.abs_pos_emb_query = nn.Embedding(seq_length, k * h)
        # self.rel_pos_score = nn.Embedding(2 * seq_length - 1, h)
        # sigma, alpha = 0.5, 1
        # x = torch.arange(2 * seq_length - 1) - seq_length
        # init_val = (alpha * (torch.exp(-((x/sigma)**2) / 2))).unsqueeze(-1).repeat(1, h)
        # self.rel_pos_score.weight.data = init_val
        # position_ids_l = torch.arange(seq_length, dtype=torch.long).view(-1, 1)
        # position_ids_r = torch.arange(seq_length, dtype=torch.long).view(1, -1)
        # self.distance = position_ids_r - position_ids_l + seq_length - 1

        position_ids_l = torch.arange(seq_length, dtype=torch.long).view(-1, 1)
        position_ids_r = torch.arange(seq_length, dtype=torch.long).view(1, -1)
        self.distance = (position_ids_r - position_ids_l).abs()
        self.window_size = 5

        self.dropout = nn.Dropout(p=0.2)
        self.silu = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.gate.data.fill_(1)

        # Heads unifier
        self.unifyheads = nn.Linear(k * h *2, k)

    def forward(self, x):

        # Extraction dimensions
        b, t, k = x.size()  # batch_size, number_of_timesteps, number_of_time_series

        # Checking Embedding dimension
        assert self.k == k, 'Number of time series ' + str(k) + ' didn t much the number of k ' + str(
            self.k) + ' in the initiaalization of the attention layer.'
        h = self.headers

        #  Transpose to see the different time series as different channels
        x = x.transpose(1, 2)
        x_padded = self.padding_opertor(x)

        # Query, Key and Value Transformations
        queries_g = self.silu(self.toqueries(x_padded).view(b, k, h, t))
        keys_g = self.silu(self.tokeys(x_padded).view(b, k, h, t))
        values_g = self.silu(self.tovalues(x).view(b, k, h, t))


        queries_l = self.silu(self.linear(x).view(b, k, h, t))
        keys_l = self.silu(self.linear(x).view(b, k, h, t))
        values_l = self.silu(self.linear(x).view(b, k, h, t))

        queries_l = queries_l.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        queries_l = queries_l.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        values_l = values_l.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        values_l = values_l.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        keys_l = keys_l.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        keys_l = keys_l.transpose(2, 3)

        queries_l = queries_l.transpose(1, 2).contiguous().view(b * h, t, k)
        keys_l = keys_l.transpose(1, 2).contiguous().view(b * h, t, k)
        values_l = values_l.transpose(1, 2).contiguous().view(b * h, t, k)


        #win local
        scores_l = torch.matmul(queries_l, keys_l.transpose(-2, -1)) / math.sqrt(queries_l.size(-1))
        # mask = None
        mask = (self.distance.to(scores_l) <= self.window_size)
        scores_l = scores_l.masked_fill(mask == 0, -1e4)


        ## initial
        # scores_l = torch.matmul(queries_l, keys_l.transpose(-2, -1)) / math.sqrt(queries_l.size(-1))
        # reweight = self.rel_pos_score(self.distance.to(scores_l)).unsqueeze(0)
        # scores_l = scores_l * (reweight / 0.1).sigmoid()
        # scores_l = scores_l.masked_fill(mask == 0, -1e4)


        # Transposition to return the canonical format
        queries_g = queries_g.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        queries_g = queries_g.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        values_g = values_g.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        values_g = values_g.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        keys_g = keys_g.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        keys_g = keys_g.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        # Weights
        queries_g = queries_g / (k ** (.25))
        keys_g = keys_g / (k ** (.25))

        queries_g = queries_g.transpose(1, 2).contiguous().view(b * h, t, k)
        keys_g = keys_g.transpose(1, 2).contiguous().view(b * h, t, k)
        values_g = values_g.transpose(1, 2).contiguous().view(b * h, t, k)

        weights = torch.bmm(queries_g, keys_g.transpose(1, 2))

        ## Mask the upper & diag of the attention matrix
        if self.mask_next:
            if self.mask_diag:
                indices = torch.triu_indices(t, t, offset=0)
                weights[:, indices[0], indices[1]] = float('-inf')
                scores_l[:, indices[0], indices[1]] = float('-inf')
            else:
                indices = torch.triu_indices(t, t, offset=1)
                weights[:, indices[0], indices[1]] = float('-inf')
                scores_l[:, indices[0], indices[1]] = float('-inf')

        # Softmax
        p_attn_l = self.dropout(F.softmax(scores_l, dim=-1))

        weights = F.softmax(weights, dim=2)

        # Output
        output_g = torch.bmm(weights, values_g)
        output_l = torch.matmul(p_attn_l, values_l)

        output_g = output_g.view(b, h, t, k)
        output_l = output_l.view(b, h, t, k)

        #gate = self.sigmoid(self.gate)
        gate = self.sigmoid(output_g)

        output_g = gate * output_g
        output_l = (1-gate) * output_l

        #output = output_g
        #output = output_l

        #output = gate * output_g +(1-gate) * output_l


        output = torch.cat([output_l, output_g], dim=1)
        output = output.transpose(1, 2).contiguous().view(b, t, k * h*2)

        return self.silu(self.unifyheads(output))  # shape (b,t,k)

# Conv Transforme Block
class ConvTransformerBLock(nn.Module):
    def __init__(self, k, headers, seq_length, kernel_size=5, mask_next=True, mask_diag=False, dropout_proba=0.2):
        super().__init__()

        # Self attention
        self.attention = SelfAttentionConv(k, headers, seq_length, kernel_size, mask_next, mask_diag)

        # First & Second Norm
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        # self.MotionBlocks = Action(k)

        # Feed Forward Network
        self.feedforward = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )
        # Dropout funtcion & Relu:
        self.dropout = nn.Dropout(p=dropout_proba)
        self.activation = nn.ReLU()

    def forward(self, x, train=False):
        # Self attention + Residual
        x = self.attention(x) + x

        # x = self.MotionBlocks(x)

        # Dropout attention
        if train:
            x = self.dropout(x)

        # First Normalization
        x = self.norm1(x)

        # Feed Froward network + residual
        x = self.feedforward(x) + x

        # Second Normalization
        x = self.norm2(x)

        return x

# Forcasting Conv Transformer :
class ForcastConvTransformer(nn.Module):
    def __init__(self, k, headers, depth, seq_length, kernel_size=5, mask_next=True, mask_diag=False, emd_dim=128,
                 dropout_proba=0.2, num_tokens=None):
        super().__init__()
        # Embedding
        self.tokens_in_count = False
        if num_tokens:
            self.tokens_in_count = True
            self.token_embedding = nn.Embedding(num_tokens, k)  # seq_length : windows

        self.hw = int(10)

        # Embedding the position
        self.position_embedding = nn.Embedding(seq_length, k)

        # Number of time series
        self.k = k  # num_of_var
        self.seq_length = seq_length
        self.emd_dim = emd_dim

        # self.gru = nn.GRU(k, 256, batch_first=True)

        # Transformer blocks
        tblocks = []
        motion = []
        for t in range(depth):
            tblocks.append(ConvTransformerBLock(k, headers, seq_length, kernel_size, mask_next, mask_diag, dropout_proba))
        self.TransformerBlocks = nn.Sequential(*tblocks)

        # for t in range(depth):
        # motion.append(Action())
        self.MotionBlocks = Action(k)

        self.gat = FeatureAttentionLayer(k, seq_length, dropout_proba, emd_dim, alpha=0.2,  use_gatv2=True, use_bias=True)

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)

        # Transformation from k dimension to numClasses
        self.topreSigma = nn.Linear(k, 1)
        self.tomu = nn.Linear(self.k*self.seq_length*3, 128)
        self.tomu1 = nn.Linear(128, k)
        self.plus = nn.Softplus()  # 激活函数，可以理解为soft relu
        self.se = SEModule(k)

    def forward(self, x, tokens=None):
        b, t, k = x.size()

        # checking that the given batch had same number of time series as the BLock had
        assert k == self.k, 'The k :' + str(
            self.k) + ' number of timeseries given in the initialization is different than what given in the x :' + str(
            k)
        assert t == self.seq_length, 'The lenght of the timeseries given t ' + str(
            t) + ' miss much with the lenght sequence given in the Tranformers initialisation self.seq_length: ' + str(
            self.seq_length)

        # x_e = self.se(x.permute(0,2,1))
        # x_e = x_e.permute(0,2,1)

        x_em = self.MotionBlocks(x)
        x_gat = self.gat(x)

        # Position embedding
        pos = torch.arange(t)
        self.pos_emb = self.position_embedding(pos).expand(b, t, k)

        # Checking token embedding
        assert self.tokens_in_count == (not (tokens is None)), 'self.tokens_in_count = ' + str(
            self.tokens_in_count) + ' should be equal to (not (tokens is None)) = ' + str((not (tokens is None)))
        if not (tokens is None):
            ## checking that the number of tockens corresponde to the number of batch elements
            assert tokens.size(0) == b
            self.tok_emb = self.token_embedding(tokens)
            self.tok_emb = self.tok_emb.expand(t, b, k).transpose(0, 1)

        # Adding Pos Embedding and token Embedding to the variable
        if not (tokens is None):
            # x_em = self.pos_emb + self.tok_emb + x
            x_em = self.pos_emb + self.tok_emb + x_em
        else:
            # x_em = self.pos_emb + x
            x_em = self.pos_emb + x_em

        # original dataset input
        if not (tokens is None):
            x_ori = self.pos_emb + self.tok_emb + x_em
        else:
            x_ori = self.pos_emb + x

        if not (tokens is None):
             x_gat = self.pos_emb + self.tok_emb + x_gat
        else:
             x_gat = self.pos_emb + x_gat

        #x = torch.clamp(x, -1, 1)

        # Transformer :
        x1 = self.TransformerBlocks(x_em)
        x2 = self.TransformerBlocks(x_ori)
        x3 = self.TransformerBlocks(x_gat)

        x1 = x1 + x_em
        x2 = x2 + x_ori
        x3 = x3 + x_gat

        x_add = x1 + x2

        #x = torch.clamp(x, -0.5, 0.5)

        x_cat = torch.cat([x1, x2, x3], 2)
        x_flatten = torch.flatten(x_cat, start_dim=1)
        # _, h = self.gru(x)
        # h = h[-1, :, :]
        mu = self.tomu(x_flatten)
        mu = self.plus(mu)

        #mu = torch.clamp(mu, -0, 1)

        mu = self.tomu1(mu)

        ### highway
        #if (self.hw > 0):

         #   z = x[:, -self.hw:, :]
         #   z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
         #   z = self.highway(z)
         #   z = z.view(-1, k)
         #   mu = mu+z

        return mu
