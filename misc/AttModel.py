# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from misc.model import AttModel
from torch.nn.parameter import Parameter
import pdb
import random


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net =  nn.Linear(self.att_hid_size, 1)
        self.min_value = -1e8
        # self.batch_norm = nn.BatchNorm1d(self.rnn_size)

    def forward(self, h, att_feats, p_att_feats):
        # The p_att_feats here is already projected
        batch_size = h.size(0)
        att_size = att_feats.numel() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                # batch * att_size * att_hid_size
        dot = F.tanh(dot)                              # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        # dot = F.dropout(dot, 0.3, training=self.training)
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size
        # att_res = self.batch_norm(att_res)

        return att_res


class Attention2(nn.Module):
    def __init__(self, opt):
        super(Attention2, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        if opt.region_attn_mode in ('add', 'mix'):
            self.alpha_net = nn.Linear(self.att_hid_size, 1)
        elif opt.region_attn_mode == 'cat':
            self.alpha_net = nn.Linear(self.att_hid_size*2, 1)
        self.min_value = -1e8
        # self.batch_norm = nn.BatchNorm1d(self.rnn_size)

    def forward(self, h, att_feats, p_att_feats, mask):
        # The p_att_feats here is already projected
        batch_size = h.size(0)
        att_size = att_feats.numel() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size

        if hasattr(self, 'alpha_net'):
            # print('Additive region attention!')
            if self.alpha_net.weight.size(1) == self.att_hid_size:
                dot = att + att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
            else:
                dot = torch.cat((xt.unsqueeze(1).expand_as(att), att_feats), 2)

            dot = F.tanh(dot)                              # batch * att_size * att_hid_size
            dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
            # dot = F.dropout(dot, 0.3, training=self.training)
            hAflat = self.alpha_net(dot)                           # (batch * att_size) * 1
        else:
            # print('Dot-product region attention!')
            assert(att.size(2) == att_h.size(1))
            hAflat = torch.matmul(att, att_h.view(batch_size, self.att_hid_size, 1))

        hAflat = hAflat.view(-1, att_size)                        # batch * att_size
        hAflat.masked_fill_(mask, self.min_value)
        
        weight = F.softmax(hAflat, dim=1)                             # batch * att_size

        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res, hAflat, att_h


class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.min_value = -1e8
        self.att_input_mode=opt.att_input_mode
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.detect_size = opt.detect_size

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1

        self.lang_lstm = nn.LSTMCell(opt.rnn_size*2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)
        self.attention2 = Attention2(opt)
        if self.att_input_mode == 'dual_region':
            self.attention2_dual = Attention2(opt)
            self.dual_pointer = nn.Sequential(nn.Linear(opt.rnn_size, 1), nn.Sigmoid())

        self.i2h_2 = nn.Linear(opt.rnn_size*2, opt.rnn_size)
        self.h2h_2 = nn.Linear(opt.rnn_size, opt.rnn_size)


    def forward(self, xt, fc_feats, conv_feats, p_conv_feats, pool_feats, p_pool_feats, att_mask, pnt_mask, state, sim_mat_static_update):
        
        att_lstm_input = torch.cat([fc_feats, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))
        if self.att_input_mode != 'region':
            att = self.attention(h_att, conv_feats, p_conv_feats)
        att2, att2_weight, att_h = self.attention2(h_att, pool_feats, p_pool_feats, att_mask[:,1:])

        max_grd_val = att2.new(pool_feats.size(0), 1).fill_(0) # dummy
        grd_val = att2.new(pool_feats.size(0), 1).fill_(0)

        if self.att_input_mode == 'both':
            lang_lstm_input = torch.cat([att+att2, h_att], 1)
        elif self.att_input_mode == 'featmap':
            lang_lstm_input = torch.cat([att, h_att], 1)
        elif self.att_input_mode == 'region':
            lang_lstm_input = torch.cat([att2, h_att], 1)
        elif self.att_input_mode == 'dual_region':
            att2_dual, _, _ = self.attention2_dual(h_att, pool_feats, p_pool_feats, att_mask[:,1:])
            dual_p = self.dual_pointer(h_att)
            lang_lstm_input = torch.cat([dual_p*att2+(1-dual_p)*att2_dual, h_att], 1)
        else:
            raise "Unknown attention input mode!"

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        output = F.dropout(h_lang, self.drop_prob_lm, self.training) # later encoded to P_{txt}^t
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state, att2_weight, att_h, max_grd_val, grd_val


class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt)
