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
import math
import pdb
import random

class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    def beam_search(self, state, rnn_output, beam_fc_feats, beam_conv_feats, beam_p_conv_feats, \
                             beam_pool_feats, beam_p_pool_feats, beam_sim_mat_static, beam_ppls, beam_pnt_mask, vis_offset, roi_offset, opt):
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt

        # def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq, \
        #                 beam_bn_seq_logprobs, beam_fg_seq, beam_fg_seq_logprobs, rnn_output, beam_pnt_mask, state):
        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_att2_ind, \
                        rnn_output, beam_pnt_mask, state, att2_ind):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys,ix = torch.sort(logprobsf,1,True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q,c]

                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_logprob, \
                                      'w':att2_ind[q] })

            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.clone() for _ in state]
            new_rnn_output = rnn_output.clone()

            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
                beam_att2_ind_prev = beam_att2_ind[:t].clone()

                beam_pnt_mask_prev = beam_pnt_mask.clone()
                beam_pnt_mask = beam_pnt_mask.clone()

            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                    beam_att2_ind[:t, vix] = beam_att2_ind_prev[:, v['q']]
                    beam_pnt_mask[:, vix] = beam_pnt_mask_prev[:, v['q']]

                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step

                new_rnn_output[vix] = rnn_output[v['q']] # dimension one is time step

                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                if t >= 1:
                    beam_att2_ind[t, vix] = v['w']
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

            state = new_state
            rnn_output = new_rnn_output

            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_att2_ind, \
                    rnn_output, state, beam_pnt_mask.t(), candidates

        # start beam search
        # opt = kwargs['opt']
        beam_size = opt.get('beam_size', 5)
        beam_att_mask = beam_pnt_mask.clone()
        rois_num = beam_ppls.size(1)

        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
        beam_att2_ind = torch.LongTensor(self.seq_length, beam_size).fill_(-1)
        att2_ind = torch.LongTensor(beam_size).fill_(-1)

        beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
        done_beams = []
        beam_pnt_mask_list = []
        beam_pnt_mask_list.append(beam_pnt_mask)

        for t in range(self.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            decoded = F.log_softmax(self.logit(rnn_output), dim=1)
            
            logprobs = decoded

            logprobsf = logprobs.data.cpu() # lets go to CPU for more efficiency in indexing operations
            # suppress UNK tokens in the decoding
            # logprobsf[:,logprobsf.size(1)-1] =  logprobsf[:, logprobsf.size(1)-1] - 1000  

            beam_seq, beam_seq_logprobs, \
            beam_logprobs_sum, beam_att2_ind, \
            rnn_output, state, beam_pnt_mask_new, \
            candidates_divm = beam_step(logprobsf,
                                        beam_size,
                                        t,
                                        beam_seq,
                                        beam_seq_logprobs,
                                        beam_logprobs_sum,
                                        beam_att2_ind,
                                        rnn_output,
                                        beam_pnt_mask_list[-1].t(),
                                        state, att2_ind)

            # encode as vectors
            it = beam_seq[t].cuda()
            assert(torch.sum(it>=self.vocab_size) == 0)

            roi_idx = it.clone() - self.vocab_size - 1 # starting from 0
            roi_mask = roi_idx < 0

            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'att2' : beam_att2_ind[:, vix],
                        'p': beam_logprobs_sum[vix],
                    }

                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # updating the mask, and make sure that same object won't happen in the caption
            pnt_idx_offset = roi_idx + roi_offset + 1
            pnt_idx_offset[roi_mask] = 0
            beam_pnt_mask = beam_pnt_mask_new.data.clone()

            beam_pnt_mask.view(-1)[pnt_idx_offset] = 1
            beam_pnt_mask.view(-1)[0] = 0
            beam_pnt_mask_list.append(Variable(beam_pnt_mask))

            xt = self.embed(Variable(it))

            rnn_output, state, att2_weight, att_h, _, _ = self.core(xt, beam_fc_feats, beam_conv_feats,
                beam_p_conv_feats, beam_pool_feats, beam_p_pool_feats, beam_att_mask, beam_pnt_mask_list[-1], \
                state, Variable(beam_pool_feats.data.new(beam_size, rois_num).fill_(0)), beam_sim_mat_static, self)
            _, att2_ind = torch.max(att2_weight, 1)

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams
