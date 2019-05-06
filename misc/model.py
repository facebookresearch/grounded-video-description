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
from torch.autograd import Variable
import math
import numpy as np
import random
import pdb
import pickle

import misc.utils as utils
from misc.CaptionModelBU import CaptionModel
from misc.transformer import Transformer, TransformerDecoder


class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.detect_size = opt.detect_size # number of object classes
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.seg_info_size = 50
        self.fc_feat_size = opt.fc_feat_size+self.seg_info_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.seq_per_img = opt.seq_per_img
        self.itod = opt.itod
        self.att_input_mode = opt.att_input_mode
        self.transfer_mode = opt.transfer_mode
        self.test_mode = opt.test_mode
        self.enable_BUTD = opt.enable_BUTD
        self.w_grd = opt.w_grd
        self.w_cls = opt.w_cls
        self.num_sampled_frm = opt.num_sampled_frm
        self.num_prop_per_frm = opt.num_prop_per_frm
        self.att_model = opt.att_model
        self.unk_idx = int(opt.wtoi['UNK'])

        if opt.region_attn_mode == 'add':
            self.alpha_net = nn.Linear(self.att_hid_size, 1)
        elif opt.region_attn_mode == 'cat':
            self.alpha_net = nn.Linear(self.att_hid_size*2, 1)

        self.stride = 32 # downsizing from input image to feature map

        self.t_attn_size = opt.t_attn_size
        self.tiny_value = 1e-8

        if self.enable_BUTD:
            assert(self.att_input_mode == 'region')
            self.pool_feat_size = self.att_feat_size
        else:
            self.pool_feat_size = self.att_feat_size+300+self.detect_size+1

        self.min_value = -1e8
        opt.beta = 1
        self.beta = opt.beta

        self.loc_fc = nn.Sequential(nn.Linear(5, 300),
                                    nn.ReLU(),
                                    nn.Dropout(inplace=True))

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size,
                                self.input_encoding_size), # det is 1-indexed
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm, inplace=True))

        if self.transfer_mode in ('none', 'cls'):
            self.vis_encoding_size = 2048
        elif self.transfer_mode == 'both':
            self.vis_encoding_size = 2348
        elif self.transfer_mode == 'glove':
            self.vis_encoding_size = 300
        else:
            raise NotImplementedError

        self.vis_embed = nn.Sequential(nn.Embedding(self.detect_size+1,
                                self.vis_encoding_size), # det is 1-indexed
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm, inplace=True)
                                )

        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm, inplace=True))

        self.seg_info_embed = nn.Sequential(nn.Linear(4, self.seg_info_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm, inplace=True))

        self.att_embed = nn.ModuleList([nn.Sequential(nn.Linear(2048, self.rnn_size//2), # for rgb feature
                                                      nn.ReLU(),
                                                      nn.Dropout(self.drop_prob_lm, inplace=True)),
                                        nn.Sequential(nn.Linear(1024, self.rnn_size//2), # for motion feature
                                                      nn.ReLU(),
                                                      nn.Dropout(self.drop_prob_lm, inplace=True))])

        self.att_embed_aux = nn.Sequential(nn.BatchNorm1d(self.rnn_size),
                                          nn.ReLU())

        self.pool_embed = nn.Sequential(nn.Linear(self.pool_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm, inplace=True))

        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ctx2pool = nn.Linear(self.rnn_size, self.att_hid_size)

        self.logit = nn.Linear(self.rnn_size, self.vocab_size)

        if opt.obj_interact:
            n_layers = 2
            n_heads = 6
            attn_drop = 0.2
            self.obj_interact = Transformer(self.rnn_size, 0, 0,
                d_hidden=int(self.rnn_size/2),
                n_layers=n_layers,
                n_heads=n_heads,
                drop_ratio=attn_drop,
                pe=False)

        if self.att_model == 'transformer':
            n_layers = 2
            n_heads = 6
            attn_drop = 0.2
            print('initiailze language decoder transformer...')
            self.cap_model = TransformerDecoder(self.rnn_size, 0, self.vocab_size, \
                d_hidden = self.rnn_size//2, n_layers=n_layers, n_heads=n_heads, drop_ratio=attn_drop)

        if opt.t_attn_mode == 'bilstm': # frame-wise feature encoding
            n_layers = 2
            attn_drop = 0.2
            self.context_enc = nn.LSTM(self.rnn_size, self.rnn_size//2, n_layers, dropout=attn_drop, \
                bidirectional=True, batch_first=True)
        elif opt.t_attn_mode == 'bigru':
            n_layers = 2
            attn_drop = 0.2
            self.context_enc = nn.GRU(self.rnn_size, self.rnn_size//2, n_layers, dropout=attn_drop, \
                bidirectional=True, batch_first=True)
        else:
            raise NotImplementedError

        self.ctx2pool_grd = nn.Sequential(nn.Linear(self.att_feat_size, self.vis_encoding_size), # fc7 layer
                                          nn.ReLU(),
                                          nn.Dropout(self.drop_prob_lm, inplace=True)
                                          )

        self.critLM = utils.LMCriterion(opt)

        # initialize the glove weight for the labels.
        # self.det_fc[0].weight.data.copy_(opt.glove_vg_cls)
        # for p in self.det_fc[0].parameters(): p.requires_grad=False

        # self.embed[0].weight.data.copy_(torch.cat((opt.glove_w, opt.glove_clss)))
        # for p in self.embed[0].parameters(): p.requires_grad=False

        # weights transfer for fc7 layer
        with open('data/detectron_weights/fc7_w.pkl') as f:
            fc7_w = torch.from_numpy(pickle.load(f))
        with open('data/detectron_weights/fc7_b.pkl') as f:
            fc7_b = torch.from_numpy(pickle.load(f))
        self.ctx2pool_grd[0].weight[:self.att_feat_size].data.copy_(fc7_w)
        self.ctx2pool_grd[0].bias[:self.att_feat_size].data.copy_(fc7_b)

        if self.transfer_mode in ('cls', 'both'):
            # find nearest neighbour class for transfer
            with open('data/detectron_weights/cls_score_w.pkl') as f:
                cls_score_w = torch.from_numpy(pickle.load(f)) # 1601x2048
            with open('data/detectron_weights/cls_score_b.pkl') as f:
                cls_score_b = torch.from_numpy(pickle.load(f)) # 1601x2048

            assert(len(opt.itod)+1 == opt.glove_clss.size(0)) # index 0 is background
            assert(len(opt.vg_cls) == opt.glove_vg_cls.size(0)) # index 0 is background

            sim_matrix = torch.matmul(opt.glove_vg_cls/torch.norm(opt.glove_vg_cls, dim=1).unsqueeze(1), \
                (opt.glove_clss/torch.norm(opt.glove_clss, dim=1).unsqueeze(1)).transpose(1,0))

            max_sim, matched_cls = torch.max(sim_matrix, dim=0)
            self.max_sim = max_sim
            self.matched_cls = matched_cls

            vis_classifiers = opt.glove_clss.new(self.detect_size+1, cls_score_w.size(1)).fill_(0)
            self.vis_classifiers_bias = nn.Parameter(opt.glove_clss.new(self.detect_size+1).fill_(0))
            vis_classifiers[0] = cls_score_w[0] # background
            self.vis_classifiers_bias[0].data.copy_(cls_score_b[0])
            for i in range(1, self.detect_size+1):
                vis_classifiers[i] = cls_score_w[matched_cls[i]]
                self.vis_classifiers_bias[i].data.copy_(cls_score_b[matched_cls[i]])
                if max_sim[i].item() < 0.9:
                    print('index: {}, similarity: {:.2}, {}, {}'.format(i, max_sim[i].item(), \
                        opt.itod[i], opt.vg_cls[matched_cls[i]]))

            if self.transfer_mode == 'cls':
                self.vis_embed[0].weight.data.copy_(vis_classifiers)
            else:
                self.vis_embed[0].weight.data.copy_(torch.cat((vis_classifiers, opt.glove_clss), dim=1))
        elif self.transfer_mode == 'glove':
            self.vis_embed[0].weight.data.copy_(opt.glove_clss)
        elif self.transfer_mode == 'none':
            print('No knowledge transfer...')
        else:
            raise NotImplementedError

        # for p in self.ctx2pool_grd.parameters(): p.requires_grad=False
        # for p in self.vis_embed[0].parameters(): p.requires_grad=False

        if opt.enable_visdom:
            import visdom
            self.vis = visdom.Visdom(server=opt.visdom_server, env='vis-'+opt.id)


    def forward(self, segs_feat, seq, gt_seq, num, ppls, gt_boxes, mask_boxes, ppls_feat, frm_mask, sample_idx, pnt_mask, opt, eval_opt = {}):
        if opt == 'MLE':
            return self._forward(segs_feat, seq, gt_seq, ppls, gt_boxes, mask_boxes, num, ppls_feat, frm_mask, sample_idx, pnt_mask)
        elif opt == 'GRD':
            return self._forward(segs_feat, seq, gt_seq, ppls, gt_boxes, mask_boxes, num, ppls_feat, frm_mask, sample_idx, pnt_mask, True)
        elif opt == 'sample':
            seq, seqLogprobs, att2, sim_mat = self._sample(segs_feat, ppls, num, ppls_feat, sample_idx, pnt_mask, eval_opt)
            return Variable(seq), Variable(att2), Variable(sim_mat)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))


    def _grounder(self, xt, att_feats, mask, bias=None):
        # xt - B, seq_cnt, enc_size
        # att_feats - B, rois_num, enc_size
        # mask - B, rois_num
        #
        # dot - B, seq_cnt, rois_num

        B, S, _ = xt.size()
        _, R, _ = att_feats.size()

        if hasattr(self, 'alpha_net'):
            # Additive attention for grounding
            if self.alpha_net.weight.size(1) == self.att_hid_size:
                dot = xt.unsqueeze(2) + att_feats.unsqueeze(1)
            else:
                dot = torch.cat((xt.unsqueeze(2).expand(B, S, R, self.att_hid_size),
                                 att_feats.unsqueeze(1).expand(B, S, R, self.att_hid_size)), 3)
            dot = F.tanh(dot)
            dot = self.alpha_net(dot).squeeze(-1)
        else:
            # Dot-product attention for grounding
            assert(xt.size(-1) == att_feats.size(-1))
            dot = torch.matmul(xt, att_feats.permute(0,2,1).contiguous()) # B, seq_cnt, rois_num

        if bias is not None:
            assert(bias.numel() == dot.numel())
            dot += bias

        if mask.dim() == 2:
            expanded_mask = mask.unsqueeze(1).expand_as(dot)
        elif mask.dim() == 3: # if expanded already
            expanded_mask = mask
        else:
            raise NotImplementedError

        dot.masked_fill_(expanded_mask, self.min_value)

        return dot


    def _forward(self, segs_feat, input_seq, gt_seq, ppls, gt_boxes, mask_boxes, num, ppls_feat, frm_mask, sample_idx, pnt_mask, eval_obj_ground=False):

        seq = gt_seq[:, :self.seq_per_img, :].clone().view(-1, gt_seq.size(2)) # choose the first seq_per_img
        seq = torch.cat((Variable(seq.data.new(seq.size(0), 1).fill_(0)), seq), 1)
        input_seq = input_seq.view(-1, input_seq.size(2), input_seq.size(3)) # B*self.seq_per_img, self.seq_length+1, 5
        input_seq_update = input_seq.data.clone()

        batch_size = segs_feat.size(0) # B
        seq_batch_size = seq.size(0) # B*self.seq_per_img
        rois_num = ppls.size(1) # max_num_proposal of the batch

        state = self.init_hidden(seq_batch_size) # self.num_layers, B*self.seq_per_img, self.rnn_size
        rnn_output = []
        roi_labels = [] # store which proposal match the gt box
        att2_weights = []
        h_att_output = []
        max_grd_output = []
        frm_mask_output = []

        conv_feats = segs_feat
        sample_idx_mask = conv_feats.new(batch_size, conv_feats.size(1), 1).fill_(1).byte()
        for i in range(batch_size):
            sample_idx_mask[i, sample_idx[i,0]:sample_idx[i,1]] = 0
        fc_feats = torch.mean(segs_feat, dim=1)
        fc_feats = torch.cat((F.layer_norm(fc_feats, [self.fc_feat_size-self.seg_info_size]), \
                              F.layer_norm(self.seg_info_embed(num[:, 3:7].float()), [self.seg_info_size])), dim=-1)

        # pooling the conv_feats
        pool_feats = ppls_feat
        pool_feats = self.ctx2pool_grd(pool_feats)
        g_pool_feats = pool_feats

        # calculate the overlaps between the rois/rois and rois/gt_bbox.
        # apply both frame mask and proposal mask
        overlaps = utils.bbox_overlaps(ppls.data, gt_boxes.data, \
                                      (frm_mask | pnt_mask[:, 1:].unsqueeze(-1)).data)

        # visual words embedding
        vis_word = Variable(torch.Tensor(range(0, self.detect_size+1)).type(input_seq.type()))
        vis_word_embed = self.vis_embed(vis_word)
        assert(vis_word_embed.size(0) == self.detect_size+1)

        p_vis_word_embed = vis_word_embed.view(1, self.detect_size+1, self.vis_encoding_size) \
            .expand(batch_size, self.detect_size+1, self.vis_encoding_size).contiguous()

        if hasattr(self, 'vis_classifiers_bias'):
            bias = self.vis_classifiers_bias.type(p_vis_word_embed.type()) \
                                                  .view(1,-1,1).expand(p_vis_word_embed.size(0), \
                                                  p_vis_word_embed.size(1), g_pool_feats.size(1))
        else:
            bias = None

        # region-class similarity matrix
        sim_mat_static = self._grounder(p_vis_word_embed, g_pool_feats, pnt_mask[:,1:], bias)
        sim_mat_static_update = sim_mat_static.view(batch_size, 1, self.detect_size+1, rois_num) \
            .expand(batch_size, self.seq_per_img, self.detect_size+1, rois_num).contiguous() \
            .view(seq_batch_size, self.detect_size+1, rois_num)
        sim_mat_static = F.softmax(sim_mat_static, dim=1)

        if self.test_mode:
            cls_pred = 0
        else:
            sim_target = utils.sim_mat_target(overlaps, gt_boxes[:,:,5].data) # B, num_box, num_rois
            sim_mask = (sim_target > 0)
            if not eval_obj_ground:
                masked_sim = torch.gather(sim_mat_static, 1, sim_target)
                masked_sim = torch.masked_select(masked_sim, sim_mask)
                cls_loss = F.binary_cross_entropy(masked_sim, masked_sim.new(masked_sim.size()).fill_(1))
            else:
                # region classification accuracy
                sim_target_masked = torch.masked_select(sim_target, sim_mask)
                sim_mat_masked = torch.masked_select(torch.max(sim_mat_static, dim=1)[1].unsqueeze(1).expand_as(sim_target), sim_mask)
                cls_pred = torch.stack((sim_target_masked, sim_mat_masked), dim=1).data

        if not self.enable_BUTD:
            loc_input = ppls.data.new(batch_size, rois_num, 5)
            loc_input[:,:,:4] = ppls.data[:,:,:4] / 720.
            loc_input[:,:,4] = ppls.data[:,:,4]*1./self.num_sampled_frm
            loc_feats = self.loc_fc(Variable(loc_input)) # encode the locations
            label_feat = sim_mat_static.permute(0,2,1).contiguous()
            pool_feats = torch.cat((F.layer_norm(pool_feats, [pool_feats.size(-1)]), \
                F.layer_norm(loc_feats, [loc_feats.size(-1)]), F.layer_norm(label_feat, [label_feat.size(-1)])), 2)

        # replicate the feature to map the seq size.
        fc_feats = fc_feats.view(batch_size, 1, self.fc_feat_size)\
                .expand(batch_size, self.seq_per_img, self.fc_feat_size)\
                .contiguous().view(-1, self.fc_feat_size)
        pool_feats = pool_feats.view(batch_size, 1, rois_num, self.pool_feat_size)\
                .expand(batch_size, self.seq_per_img, rois_num, self.pool_feat_size)\
                .contiguous().view(-1, rois_num, self.pool_feat_size)
        g_pool_feats = g_pool_feats.view(batch_size, 1, rois_num, self.vis_encoding_size) \
                .expand(batch_size, self.seq_per_img, rois_num, self.vis_encoding_size) \
                .contiguous().view(-1, rois_num, self.vis_encoding_size)
        pnt_mask = pnt_mask.view(batch_size, 1, rois_num+1).expand(batch_size, self.seq_per_img, rois_num+1)\
                .contiguous().view(-1, rois_num+1)
        overlaps = overlaps.view(batch_size, 1, rois_num, overlaps.size(2)) \
                .expand(batch_size, self.seq_per_img, rois_num, overlaps.size(2)) \
                .contiguous().view(-1, rois_num, overlaps.size(2))

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        pool_feats = self.pool_embed(pool_feats)

        # object region interactions
        if hasattr(self, 'obj_interact'):
            pool_feats = self.obj_interact(pool_feats)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_pool_feats = self.ctx2pool(pool_feats) # same here

        if self.att_input_mode in ('both', 'featmap'):
            conv_feats_splits = torch.split(conv_feats, 2048, 2)
            conv_feats = torch.cat([m(c) for (m,c) in zip(self.att_embed, conv_feats_splits)], dim=2)
            conv_feats = conv_feats.permute(0,2,1).contiguous() # inconsistency between Torch TempConv and PyTorch Conv1d
            conv_feats = self.att_embed_aux(conv_feats)
            conv_feats = conv_feats.permute(0,2,1).contiguous() # inconsistency between Torch TempConv and PyTorch Conv1d
            conv_feats = self.context_enc(conv_feats)[0]

            conv_feats = conv_feats.masked_fill(sample_idx_mask, 0)
            conv_feats = conv_feats.view(batch_size, 1, self.t_attn_size, self.rnn_size)\
                .expand(batch_size, self.seq_per_img, self.t_attn_size, self.rnn_size)\
                .contiguous().view(-1, self.t_attn_size, self.rnn_size)
            p_conv_feats = self.ctx2att(conv_feats) # self.rnn_size (1024) -> self.att_hid_size (512)
        else:
            # dummy
            conv_feats = pool_feats.new(1,1).fill_(0)
            p_conv_feats = pool_feats.new(1,1).fill_(0)

        if self.att_model == 'transformer': # Masked Transformer does not support box supervision yet
            if self.att_input_mode == 'both':
                lm_loss = self.cap_model([conv_feats, pool_feats], seq)
            elif self.att_input_mode == 'featmap':
                lm_loss = self.cap_model([conv_feats, conv_feats], seq)
            elif self.att_input_mode == 'region':
                lm_loss = self.cap_model([pool_feats, pool_feats], seq)
            return lm_loss.unsqueeze(0), lm_loss.new(1).fill_(0), lm_loss.new(1).fill_(0), \
                lm_loss.new(1).fill_(0), lm_loss.new(1).fill_(0), lm_loss.new(1).fill_(0)
        elif self.att_model == 'topdown':
            for i in range(self.seq_length):
                it = seq[:, i].clone()

                # break if all the sequences end
                if i >= 1 and seq[:, i].data.sum() == 0:
                    break

                xt = self.embed(it)

                if not eval_obj_ground:
                    roi_label = utils.bbox_target(mask_boxes[:,:,:,i+1], overlaps, input_seq[:,i+1], \
                        input_seq_update[:,i+1], self.vocab_size) # roi_label if for the target seq
                    roi_labels.append(roi_label.view(seq_batch_size, -1))

                    # use frame mask during training
                    box_mask = mask_boxes[:,0,:,i+1].contiguous().unsqueeze(1).expand((
                        batch_size, rois_num, mask_boxes.size(2)))
                    frm_mask_on_prop = (torch.sum((~(box_mask | frm_mask)), dim=2)<=0)
                    frm_mask_on_prop = torch.cat((frm_mask_on_prop.new(batch_size, 1).fill_(0.), \
                        frm_mask_on_prop), dim=1) | pnt_mask
                    output, state, att2_weight, att_h, max_grd_val, grd_val = self.core(xt, fc_feats, \
                        conv_feats, p_conv_feats, pool_feats, p_pool_feats, pnt_mask, frm_mask_on_prop, \
                        state, sim_mat_static_update)
                    frm_mask_output.append(frm_mask_on_prop)
                else:
                    output, state, att2_weight, att_h, max_grd_val, grd_val = self.core(xt, fc_feats, \
                        conv_feats, p_conv_feats, pool_feats, p_pool_feats, pnt_mask, pnt_mask, \
                        state, sim_mat_static_update)

                att2_weights.append(att2_weight)
                h_att_output.append(att_h) # the hidden state of attention LSTM
                rnn_output.append(output)
                max_grd_output.append(max_grd_val)

            seq_cnt = len(rnn_output)
            rnn_output = torch.cat([_.unsqueeze(1) for _ in rnn_output], 1) # seq_batch_size, seq_cnt, vocab
            h_att_output = torch.cat([_.unsqueeze(1) for _ in h_att_output], 1)
            att2_weights = torch.cat([_.unsqueeze(1) for _ in att2_weights], 1) # seq_batch_size, seq_cnt, att_size
            max_grd_output = torch.cat([_.unsqueeze(1) for _ in max_grd_output], 1)
            if not eval_obj_ground:
                frm_mask_output = torch.cat([_.unsqueeze(1) for _ in frm_mask_output], 1)
                roi_labels = torch.cat([_.unsqueeze(1) for _ in roi_labels], 1)

            decoded = F.log_softmax(self.beta * self.logit(rnn_output), dim=2) # text word prob
            decoded  = decoded.view((seq_cnt)*seq_batch_size, -1)

            # object grounding
            h_att_all = h_att_output # hidden states from the Attention LSTM
            xt_clamp = torch.clamp(input_seq[:, 1:seq_cnt+1, 0].clone()-self.vocab_size, min=0)
            xt_all = self.vis_embed(xt_clamp)

            if hasattr(self, 'vis_classifiers_bias'):
                bias = self.vis_classifiers_bias[xt_clamp].type(xt_all.type()) \
                                                .unsqueeze(2).expand(seq_batch_size, seq_cnt, rois_num)
            else:
                bias = 0

            if not eval_obj_ground:
                # att2_weights/ground_weights with both proposal mask and frame mask
                ground_weights = self._grounder(xt_all, g_pool_feats, frm_mask_output[:,:,1:], bias+att2_weights)
                lm_loss, att2_loss, ground_loss = self.critLM(decoded, att2_weights, ground_weights, \
                    seq[:, 1:seq_cnt+1].clone(), roi_labels[:, :seq_cnt, :].clone(), input_seq[:, 1:seq_cnt+1, 0].clone())
                return lm_loss.unsqueeze(0), att2_loss.unsqueeze(0), ground_loss.unsqueeze(0), cls_loss.unsqueeze(0)
            else:
                # att2_weights/ground_weights with proposal mask only
                ground_weights = self._grounder(xt_all, g_pool_feats, pnt_mask[:,1:], bias+att2_weights)
                return cls_pred, torch.max(att2_weights.view(seq_batch_size, seq_cnt, self.num_sampled_frm, \
                    self.num_prop_per_frm), dim=-1)[1], torch.max(ground_weights.view(seq_batch_size, \
                    seq_cnt, self.num_sampled_frm, self.num_prop_per_frm), dim=-1)[1]


    def _sample(self, segs_feat, ppls, num, ppls_feat, sample_idx, pnt_mask, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        inference_mode = opt.get('inference_mode', True)

        batch_size = segs_feat.size(0)
        rois_num = ppls.size(1)

        if beam_size > 1:
            return self._sample_beam(segs_feat, ppls, num, ppls_feat, sample_idx, pnt_mask, opt)

        conv_feats = segs_feat
        sample_idx_mask = conv_feats.new(batch_size, conv_feats.size(1), 1).fill_(1).byte()
        for i in range(batch_size):
            sample_idx_mask[i, sample_idx[i,0]:sample_idx[i,1]] = 0
        fc_feats = torch.mean(segs_feat, dim=1)
        fc_feats = torch.cat((F.layer_norm(fc_feats, [self.fc_feat_size-self.seg_info_size]), \
                              F.layer_norm(self.seg_info_embed(num[:, 3:7].float()), [self.seg_info_size])), dim=-1)

        pool_feats = ppls_feat
        pool_feats = self.ctx2pool_grd(pool_feats)
        g_pool_feats = pool_feats

        att_mask = pnt_mask.clone()

        # visual words embedding
        vis_word = Variable(torch.Tensor(range(0, self.detect_size+1)).type(fc_feats.type())).long()
        vis_word_embed = self.vis_embed(vis_word)
        assert(vis_word_embed.size(0) == self.detect_size+1)

        p_vis_word_embed = vis_word_embed.view(1, self.detect_size+1, self.vis_encoding_size) \
            .expand(batch_size, self.detect_size+1, self.vis_encoding_size).contiguous()

        if hasattr(self, 'vis_classifiers_bias'):
            bias = self.vis_classifiers_bias.type(p_vis_word_embed.type()) \
                                                  .view(1,-1,1).expand(p_vis_word_embed.size(0), \
                                                  p_vis_word_embed.size(1), g_pool_feats.size(1))
        else:
            bias = None

        sim_mat_static = self._grounder(p_vis_word_embed, g_pool_feats, pnt_mask[:,1:], bias)
        sim_mat_static_update = sim_mat_static
        sim_mat_static = F.softmax(sim_mat_static, dim=1)

        if not self.enable_BUTD:
            loc_input = ppls.data.new(batch_size, rois_num, 5)
            loc_input[:,:,:4] = ppls.data[:,:,:4] / 720.
            loc_input[:,:,4] = ppls.data[:,:,4]*1./self.num_sampled_frm
            loc_feats = self.loc_fc(Variable(loc_input)) # encode the locations
            label_feat = sim_mat_static.permute(0,2,1).contiguous()
            pool_feats = torch.cat((F.layer_norm(pool_feats, [pool_feats.size(-1)]), F.layer_norm(loc_feats, \
                [loc_feats.size(-1)]), F.layer_norm(label_feat, [label_feat.size(-1)])), 2)

        # embed fc and att feats
        pool_feats = self.pool_embed(pool_feats)
        fc_feats = self.fc_embed(fc_feats)
        # object region interactions
        if hasattr(self, 'obj_interact'):
            pool_feats = self.obj_interact(pool_feats)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_pool_feats = self.ctx2pool(pool_feats)

        if self.att_input_mode in ('both', 'featmap'):
            conv_feats_splits = torch.split(conv_feats, 2048, 2)
            conv_feats = torch.cat([m(c) for (m,c) in zip(self.att_embed, conv_feats_splits)], dim=2)
            conv_feats = conv_feats.permute(0,2,1).contiguous() # inconsistency between Torch TempConv and PyTorch Conv1d
            conv_feats = self.att_embed_aux(conv_feats)
            conv_feats = conv_feats.permute(0,2,1).contiguous() # inconsistency between Torch TempConv and PyTorch Conv1d
            conv_feats = self.context_enc(conv_feats)[0]

            conv_feats = conv_feats.masked_fill(sample_idx_mask, 0)
            p_conv_feats = self.ctx2att(conv_feats)
        else:
            conv_feats = pool_feats.new(1,1).fill_(0)
            p_conv_feats = pool_feats.new(1,1).fill_(0)

        if self.att_model == 'transformer':
            if self.att_input_mode == 'both':
                seq = self.cap_model([conv_feats, pool_feats], [], infer=True, seq_length=self.seq_length)
            elif self.att_input_mode == 'featmap':
                seq = self.cap_model([conv_feats, conv_feats], [], infer=True, seq_length=self.seq_length)
            elif self.att_input_mode == 'region':
                seq = self.cap_model([pool_feats, pool_feats], [], infer=True, seq_length=self.seq_length)

            return seq, seq.new(batch_size, 1).fill_(0), seq.new(batch_size, 1).fill_(0).long()
        elif self.att_model == 'topdown':
            state = self.init_hidden(batch_size)

            seq = []
            seqLogprobs = []
            att2_weights = []

            for t in range(self.seq_length + 1):
                if t == 0: # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                elif sample_max:
                    sampleLogprobs_tmp, it_tmp = torch.topk(logprobs.data, 2, dim=1)
                    unk_mask = (it_tmp[:,0] != self.unk_idx) # mask on non-unk
                    sampleLogprobs = unk_mask.float()*sampleLogprobs_tmp[:,0] + (1-unk_mask.float())*sampleLogprobs_tmp[:,1]
                    it = unk_mask.long()*it_tmp[:,0] + (1-unk_mask.long())*it_tmp[:,1]
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, Variable(it)) # gather the logprobs at sampled positions
                    it = it.view(-1).long() # and flatten indices for downstream processing

                xt = self.embed(Variable(it))
                if t >= 1:
                    seq.append(it) #seq[t] the input of t+2 time step
                    seqLogprobs.append(sampleLogprobs.view(-1))

                if t < self.seq_length:
                    rnn_output, state, att2_weight, att_h, _, _ = self.core(xt, fc_feats, conv_feats, \
                        p_conv_feats, pool_feats, p_pool_feats, att_mask, pnt_mask, state, \
                        sim_mat_static_update)

                    decoded = F.log_softmax(self.beta * self.logit(rnn_output), dim=1)

                    logprobs = decoded
                    att2_weights.append(att2_weight)

            seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
            seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
            att2_weights = torch.cat([_.unsqueeze(1) for _ in att2_weights], 1) # batch_size, seq_cnt, att_size

            return seq, seqLogprobs, att2_weights, sim_mat_static


    def _sample_beam(self, segs_feat, ppls, num, ppls_feat, sample_idx, pnt_mask, opt={}):

        batch_size = ppls.size(0)
        rois_num = ppls.size(1)

        beam_size = opt.get('beam_size', 10)

        conv_feats = segs_feat
        sample_idx_mask = conv_feats.new(batch_size, conv_feats.size(1), 1).fill_(1).byte()
        for i in range(batch_size):
            sample_idx_mask[i, sample_idx[i,0]:sample_idx[i,1]] = 0
        fc_feats = torch.mean(segs_feat, dim=1)
        fc_feats = torch.cat((F.layer_norm(fc_feats, [self.fc_feat_size-self.seg_info_size]), \
                              F.layer_norm(self.seg_info_embed(num[:, 3:7].float()), [self.seg_info_size])), dim=-1)

        pool_feats = ppls_feat
        pool_feats = self.ctx2pool_grd(pool_feats)
        g_pool_feats = pool_feats

        # visual words embedding
        vis_word = Variable(torch.Tensor(range(0, self.detect_size+1)).type(fc_feats.type())).long()
        vis_word_embed = self.vis_embed(vis_word)
        assert(vis_word_embed.size(0) == self.detect_size+1)

        p_vis_word_embed = vis_word_embed.view(1, self.detect_size+1, self.vis_encoding_size) \
            .expand(batch_size, self.detect_size+1, self.vis_encoding_size).contiguous()

        if hasattr(self, 'vis_classifiers_bias'):
            bias = self.vis_classifiers_bias.type(p_vis_word_embed.type()) \
                                                  .view(1,-1,1).expand(p_vis_word_embed.size(0), \
                                                  p_vis_word_embed.size(1), g_pool_feats.size(1))
        else:
            bias = None

        sim_mat_static = self._grounder(p_vis_word_embed, g_pool_feats, pnt_mask[:,1:], bias)
        sim_mat_static_update = sim_mat_static
        sim_mat_static = F.softmax(sim_mat_static, dim=1)

        if not self.enable_BUTD:
            loc_input = ppls.data.new(batch_size, rois_num, 5)
            loc_input[:,:,:4] = ppls.data[:,:,:4] / 720.
            loc_input[:,:,4] = ppls.data[:,:,4]*1./self.num_sampled_frm
            loc_feats = self.loc_fc(Variable(loc_input)) # encode the locations

            label_feat = sim_mat_static.permute(0,2,1).contiguous()

            pool_feats = torch.cat((F.layer_norm(pool_feats, [pool_feats.size(-1)]), F.layer_norm(loc_feats, [loc_feats.size(-1)]), \
                                    F.layer_norm(label_feat, [label_feat.size(-1)])), 2)

        # embed fc and att feats
        pool_feats = self.pool_embed(pool_feats)
        fc_feats = self.fc_embed(fc_feats)
        # object region interactions
        if hasattr(self, 'obj_interact'):
            pool_feats = self.obj_interact(pool_feats)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_pool_feats = self.ctx2pool(pool_feats)

        if self.att_input_mode in ('both', 'featmap'):
            conv_feats_splits = torch.split(conv_feats, 2048, 2)
            conv_feats = torch.cat([m(c) for (m,c) in zip(self.att_embed, conv_feats_splits)], dim=2)
            conv_feats = conv_feats.permute(0,2,1).contiguous() # inconsistency between Torch TempConv and PyTorch Conv1d
            conv_feats = self.att_embed_aux(conv_feats)
            conv_feats = conv_feats.permute(0,2,1).contiguous() # inconsistency between Torch TempConv and PyTorch Conv1d
            conv_feats = self.context_enc(conv_feats)[0]

            conv_feats = conv_feats.masked_fill(sample_idx_mask, 0)
            p_conv_feats = self.ctx2att(conv_feats)
        else:
            conv_feats = pool_feats.new(1,1).fill_(0)
            p_conv_feats = pool_feats.new(1,1).fill_(0)

        vis_offset = (torch.arange(0, beam_size)*rois_num).view(beam_size).type_as(ppls.data).long()
        roi_offset = (torch.arange(0, beam_size)*(rois_num+1)).view(beam_size).type_as(ppls.data).long()

        seq = ppls.data.new(self.seq_length, batch_size).zero_().long()
        seqLogprobs = ppls.data.new(self.seq_length, batch_size).float()
        att2 = ppls.data.new(self.seq_length, batch_size).fill_(-1).long()

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            beam_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            beam_pool_feats = pool_feats[k:k+1].expand(beam_size, rois_num, self.rnn_size).contiguous()
            if self.att_input_mode in ('both', 'featmap'):
                beam_conv_feats = conv_feats[k:k+1].expand(beam_size, conv_feats.size(1), self.rnn_size).contiguous()
                beam_p_conv_feats = p_conv_feats[k:k+1].expand(beam_size, conv_feats.size(1), self.att_hid_size).contiguous()
            else:
                beam_conv_feats = beam_pool_feats.new(1,1).fill_(0)
                beam_p_conv_feats = beam_pool_feats.new(1,1).fill_(0)
            beam_p_pool_feats = p_pool_feats[k:k+1].expand(beam_size, rois_num, self.att_hid_size).contiguous()

            beam_ppls = ppls[k:k+1].expand(beam_size, rois_num, 7).contiguous()
            beam_pnt_mask = pnt_mask[k:k+1].expand(beam_size, rois_num+1).contiguous()

            it = fc_feats.data.new(beam_size).long().zero_()
            xt = self.embed(Variable(it))

            beam_sim_mat_static_update = sim_mat_static_update[k:k+1].expand(beam_size, self.detect_size+1, rois_num)

            rnn_output, state, att2_weight, att_h, _, _ = self.core(xt, beam_fc_feats, beam_conv_feats,
                beam_p_conv_feats, beam_pool_feats, beam_p_pool_feats, beam_pnt_mask, beam_pnt_mask,
                state, beam_sim_mat_static_update)

            assert(att2_weight.size(0) == beam_size)
            att2[0, k] = torch.max(att2_weight, 1)[1][0]

            self.done_beams[k] = self.beam_search(state, rnn_output, beam_fc_feats, beam_conv_feats, beam_p_conv_feats, \
                                                  beam_pool_feats, beam_p_pool_feats, beam_sim_mat_static_update, beam_ppls, beam_pnt_mask, vis_offset, roi_offset, opt)
                
            seq[:, k] = self.done_beams[k][0]['seq'].cuda() # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps'].cuda()
            att2[1:, k] = self.done_beams[k][0]['att2'][1:].cuda()

        return seq.t(), seqLogprobs.t(), att2.t()
