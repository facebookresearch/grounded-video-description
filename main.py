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
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import random
import time
import os
import pickle
import torch.backends.cudnn as cudnn
import yaml
import copy
import json
import math

import opts
from misc import utils, AttModel
from collections import defaultdict

import torchvision.transforms as transforms
import pdb

# hack to allow the imports of evaluation repos
_SCRIPTPATH_ = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.join(_SCRIPTPATH_, 'tools/densevid_eval'))
sys.path.insert(0, os.path.join(_SCRIPTPATH_, 'tools/densevid_eval/coco-caption'))
sys.path.insert(0, os.path.join(_SCRIPTPATH_, 'tools/anet_entities/scripts'))

from evaluate import ANETcaptions
from eval_grd_anet_entities import ANetGrdEval


# visualization over generated sentences
def vis_infer(seg_show, seg_id, caption, att2_weights, proposals, num_box, gt_bboxs, sim_mat, seg_dim_info):
    cap = caption.split()
    output = []
    top_k_prop = 1 # plot the top 1 proposal only
    proposal = proposals[:num_box[1].item()]
    gt_bbox = gt_bboxs[:num_box[2].item()]

    sim_mat_val, sim_mat_ind = torch.max(sim_mat, dim=0)

    for j in range(len(cap)):

            max_att2_weight, top_k_alpha_idx = torch.max(att2_weights[j], dim=0)

            idx = top_k_alpha_idx
            target_frm = int(proposal[idx, 4].item())
            seg = copy.deepcopy(seg_show[target_frm, :seg_dim_info[0], :seg_dim_info[1]].numpy())
            seg_text = np.ones((67, int(seg_dim_info[1]), 3))*255
            cv2.putText(seg_text, '%s' % (cap[j]), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 0, 0), thickness=3)

            # draw the proposal box and text
            idx = top_k_alpha_idx
            bbox = proposal[idx, :4]
            bbox = tuple(int(np.round(x)) for x in proposal[idx, :4])
            class_name = opt.itod.get(sim_mat_ind[idx].item(), '__background__')
            cv2.rectangle(seg, bbox[0:2], bbox[2:4],
                         (0, 255, 0), 2)
            cv2.putText(seg, '%s: (%.2f)' % (class_name, sim_mat_val[idx]),
                       (bbox[0], bbox[1] + 25), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), thickness=2)

            output.append(np.concatenate([seg_text, seg], axis=0))

    output = np.concatenate(output, axis=1)
    if not os.path.isdir('./vis'):
        os.mkdir('./vis')
    if not os.path.isdir('./vis/'+opt.id):
        os.mkdir('./vis/'+opt.id)
    # print('Visualize segment {} and the generated sentence!'.format(seg_id))
    cv2.imwrite('./vis/'+opt.id+'/'+str(seg_id)+'_generated_sent.jpg', output[:,:,::-1])


# compute localization (attention/grounding) accuracy over GT sentences
def eval_grounding(opt, vis=None):
    model.eval()

    data_iter = iter(dataloader_val)
    cls_pred_lst = []
    cls_accu_score = defaultdict(list)
    att2_output = defaultdict(dict)
    grd_output = defaultdict(dict)
    vocab_in_split = set()

    for step in range(len(dataloader_val)):
        data = data_iter.next()
        seg_feat, iseq, gts_seq, num, proposals, bboxs, box_mask, seg_id, region_feat, frm_mask, sample_idx, ppl_mask = data

        proposals = proposals[:,:max(int(max(num[:,1])),1),:]
        ppl_mask = ppl_mask[:,:max(int(max(num[:,1])),1)]
        assert(max(int(max(num[:,1])),1) == opt.num_sampled_frm*opt.num_prop_per_frm)
        bboxs = bboxs[:,:max(int(max(num[:,2])),1),:]
        frm_mask = frm_mask[:, :max(int(max(num[:,1])),1), :max(int(max(num[:,2])),1)]
        region_feat = region_feat[:,:max(int(max(num[:,1])),1),:]

        segs_feat.resize_(seg_feat.size()).data.copy_(seg_feat)
        input_seqs.resize_(iseq.size()).data.copy_(iseq)
        gt_seqs.resize_(gts_seq.size()).data.copy_(gts_seq)
        input_num.resize_(num.size()).data.copy_(num)
        input_ppls.resize_(proposals.size()).data.copy_(proposals)
        mask_ppls.resize_(ppl_mask.size()).data.copy_(ppl_mask)
        pnt_mask = torch.cat((mask_ppls.new(mask_ppls.size(0), 1).fill_(0), mask_ppls), dim=1)
        gt_bboxs.resize_(bboxs.size()).data.copy_(bboxs) # for region cls eval only
        mask_frms.resize_(frm_mask.size()).data.copy_(frm_mask) # for region cls eval only
        ppls_feat.resize_(region_feat.size()).data.copy_(region_feat)
        sample_idx = Variable(sample_idx.type(input_seqs.type()))

        dummy = input_ppls.new(input_ppls.size(0)).byte().fill_(0)

        # cls_pred_hm_lst contains a list of tuples (clss_ind, hit/1 or miss/0)
        cls_pred_hm_lst, att2_ind, grd_ind = model(segs_feat, input_seqs, gt_seqs, input_num,
            input_ppls, gt_bboxs, dummy, ppls_feat, mask_frms, sample_idx, pnt_mask, 'GRD')

        # save attention/grounding results on GT sentences
        obj_mask = (input_seqs[:,0,1:,0] > opt.vocab_size) # Bx20
        obj_bbox_att2 = torch.gather(input_ppls.view(-1, opt.num_sampled_frm, opt.num_prop_per_frm, 7) \
            .permute(0, 2, 1, 3).contiguous(), 1, att2_ind.unsqueeze(-1).expand((att2_ind.size(0), \
            att2_ind.size(1), opt.num_sampled_frm, 7))) # Bx20x10x7
        obj_bbox_grd = torch.gather(input_ppls.view(-1, opt.num_sampled_frm, opt.num_prop_per_frm, 7) \
            .permute(0, 2, 1, 3).contiguous(), 1, grd_ind.unsqueeze(-1).expand((grd_ind.size(0), \
            grd_ind.size(1), opt.num_sampled_frm, 7))) # Bx20x10x7

        for i in range(obj_mask.size(0)):
            vid_id, seg_idx = seg_id[i].split('_segment_')
            seg_idx = str(int(seg_idx))
            tmp_result_grd = {'clss':[], 'idx_in_sent':[], 'bbox_for_all_frames':[]}
            tmp_result_att2 = {'clss':[], 'idx_in_sent':[], 'bbox_for_all_frames':[]}
            for j in range(obj_mask.size(1)):
                if obj_mask[i, j]:
                    cls_name = opt.itod[input_seqs[i,0,j+1,0].item()-opt.vocab_size]
                    vocab_in_split.update([cls_name])
                    tmp_result_att2['clss'].append(cls_name)
                    tmp_result_att2['idx_in_sent'].append(j)
                    tmp_result_att2['bbox_for_all_frames'].append(obj_bbox_att2[i, j, :, :4].tolist())
                    tmp_result_grd['clss'].append(cls_name)
                    tmp_result_grd['idx_in_sent'].append(j)
                    tmp_result_grd['bbox_for_all_frames'].append(obj_bbox_grd[i, j, :, :4].tolist())
            att2_output[vid_id][seg_idx] = tmp_result_att2
            grd_output[vid_id][seg_idx] = tmp_result_grd

        cls_pred_lst.append(cls_pred_hm_lst)

    # write results to file
    attn_file = 'results/attn-gt-sent-results-'+opt.val_split+'-'+opt.id+'.json'
    with open(attn_file, 'w') as f:
        json.dump({'results':att2_output, 'eval_mode':'GT', 'external_data':{'used':True, 'details':'Object detector pre-trained on Visual Genome on object detection task.'}}, f)
    grd_file = 'results/grd-gt-sent-results-'+opt.val_split+'-'+opt.id+'.json'
    with open(grd_file, 'w') as f:
        json.dump({'results':grd_output, 'eval_mode':'GT', 'external_data':{'used':True, 'details':'Object detector pre-trained on Visual Genome on object detection task.'}}, f)

    if not opt.test_mode:
        cls_pred_lst = torch.cat(cls_pred_lst, dim=0).cpu()
        cls_accu_lst = torch.cat((cls_pred_lst[:, 0:1], (cls_pred_lst[:, 0:1] == cls_pred_lst[:, 1:2]).long()), dim=1)
        for i in range(cls_accu_lst.size(0)):
            cls_accu_score[cls_accu_lst[i,0].long().item()].append(cls_accu_lst[i,1].item())
        print('Total number of object classes in the split: {}. {} have classification results.'.format(len(vocab_in_split), len(cls_accu_score)))
        cls_accu = np.sum([sum(hm)*1./len(hm) for i,hm in cls_accu_score.items()])*1./len(vocab_in_split)

        # offline eval
        evaluator = ANetGrdEval(reference_file=opt.grd_reference, submission_file=attn_file,
                                  split_file=opt.split_file, val_split=[opt.val_split],
                                  iou_thresh=0.5)

        attn_accu = evaluator.gt_grd_eval()
        evaluator.import_sub(grd_file)
        grd_accu = evaluator.gt_grd_eval()

        print('\nResults Summary (GT sent):')
        print('The averaged attention / grounding box accuracy across all classes is: {:.4f} / {:.4f}'.format(attn_accu, grd_accu))
        print('The averaged classification accuracy across all classes is: {:.4f}\n'.format(cls_accu))

        return attn_accu, grd_accu, cls_accu
    else:
        print('*'*62)
        print('*  [WARNING] Grounding eval unavailable for the test set!\
    *\n*            Please submit your result files under directory *\
     \n*            results/ to the eval server!                    *')
        print('*'*62)

        return 0, 0, 0


def train(epoch, opt, vis=None, vis_window=None):
    model.train()

    data_iter = iter(dataloader)
    nbatches = len(dataloader)
    train_loss = []

    lm_loss_temp = []
    att2_loss_temp = []
    ground_loss_temp = []
    cls_loss_temp = []
    start = time.time()

    for step in range(len(dataloader)-1):
        data = data_iter.next()
        seg_feat, iseq, gts_seq, num, proposals, bboxs, box_mask, seg_id, region_feat, frm_mask, sample_idx, ppl_mask = data
        proposals = proposals[:,:max(int(max(num[:,1])),1),:]
        ppl_mask = ppl_mask[:,:max(int(max(num[:,1])),1)]
        bboxs = bboxs[:,:max(int(max(num[:,2])),1),:]
        box_mask = box_mask[:,:,:max(int(max(num[:,2])),1),:]
        frm_mask = frm_mask[:,:max(int(max(num[:,1])),1),:max(int(max(num[:,2])),1)]
        region_feat = region_feat[:,:max(int(max(num[:,1])),1),:]

        segs_feat.resize_(seg_feat.size()).data.copy_(seg_feat)
        input_seqs.resize_(iseq.size()).data.copy_(iseq)
        gt_seqs.resize_(gts_seq.size()).data.copy_(gts_seq)
        input_num.resize_(num.size()).data.copy_(num)
        input_ppls.resize_(proposals.size()).data.copy_(proposals)
        mask_ppls.resize_(ppl_mask.size()).data.copy_(ppl_mask)
        # pad 1 column from a legacy reason
        pnt_mask = torch.cat((mask_ppls.new(mask_ppls.size(0), 1).fill_(0), mask_ppls), dim=1)
        gt_bboxs.resize_(bboxs.size()).data.copy_(bboxs)
        mask_bboxs.resize_(box_mask.size()).data.copy_(box_mask)
        mask_frms.resize_(frm_mask.size()).data.copy_(frm_mask)
        ppls_feat.resize_(region_feat.size()).data.copy_(region_feat)
        sample_idx = Variable(sample_idx.type(input_seqs.type()))

        loss = 0
        lm_loss, att2_loss, ground_loss, cls_loss = model(segs_feat, input_seqs, gt_seqs, input_num,
            input_ppls, gt_bboxs, mask_bboxs, ppls_feat, mask_frms, sample_idx, pnt_mask, 'MLE')

        w_att2, w_grd, w_cls = opt.w_att2, opt.w_grd, opt.w_cls
        att2_loss = w_att2*att2_loss.sum()
        ground_loss = w_grd*ground_loss.sum()
        cls_loss = w_cls*cls_loss.sum()

        if not opt.disable_caption:
            loss += lm_loss.sum()
        else:
            lm_loss.fill_(0)

        if w_att2:
            loss += att2_loss
        if w_grd:
            loss += ground_loss
        if w_cls:
            loss += cls_loss

        loss = loss / lm_loss.numel()
        train_loss.append(loss.item())

        lm_loss_temp.append(lm_loss.sum().item() / lm_loss.numel())
        att2_loss_temp.append(att2_loss.sum().item() / lm_loss.numel())
        ground_loss_temp.append(ground_loss.sum().item() / lm_loss.numel())
        cls_loss_temp.append(cls_loss.sum().item() / lm_loss.numel())

        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        if step % opt.disp_interval == 0 and step != 0:
            end = time.time()

            print("step {}/{} (epoch {}), lm_loss = {:.3f}, att2_loss = {:.3f}, ground_loss = {:.3f},cls_los = {:.3f}, lr = {:.5f}, time/batch = {:.3f}" \
                .format(step, len(dataloader), epoch, np.mean(lm_loss_temp), np.mean(att2_loss_temp), \
                        np.mean(ground_loss_temp), np.mean(cls_loss_temp), opt.learning_rate, end - start))
            start = time.time()

        if opt.enable_visdom:
            if vis_window['iter'] is None:
                vis_window['iter'] = vis.line(
                    X=np.tile(np.arange(epoch*nbatches+step, epoch*nbatches+step+1),
                              (5,1)).T,
                    Y=np.column_stack((np.asarray(np.mean(train_loss)),
                                       np.asarray(np.mean(lm_loss_temp)),
                                       np.asarray(np.mean(att2_loss_temp)),
                                       np.asarray(np.mean(ground_loss_temp)),
                                       np.asarray(np.mean(cls_loss_temp)))),
                    opts=dict(title='Training Loss',
                              xlabel='Training Iteration',
                              ylabel='Loss',
                              legend=['total', 'lm', 'attn', 'grd', 'cls'])
                )
            else:
                vis.line(
                    X=np.tile(np.arange(epoch*nbatches+step, epoch*nbatches+step+1),
                              (5,1)).T,
                    Y=np.column_stack((np.asarray(np.mean(train_loss)),
                                       np.asarray(np.mean(lm_loss_temp)),
                                       np.asarray(np.mean(att2_loss_temp)),
                                       np.asarray(np.mean(ground_loss_temp)),
                                       np.asarray(np.mean(cls_loss_temp)))),
                    opts=dict(title='Training Loss',
                              xlabel='Training Iteration',
                              ylabel='Loss',
                              legend=['total', 'lm', 'attn', 'grd', 'cls']),
                    win=vis_window['iter'],
                    update='append'
                )

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            loss_history[iteration] = loss.item()
            lr_history[iteration] = opt.learning_rate


def eval(epoch, opt, vis=None, vis_window=None):
    model.eval()

    data_iter_val = iter(dataloader_val)
    start = time.time()

    num_show = 0
    predictions = defaultdict(list)
    count = 0
    raw_caption_file = json.load(open(opt.input_raw_cap))
    min_value = -1e8

    if opt.eval_obj_grounding:
        grd_output = defaultdict(dict)

        lemma_det_dict = {opt.wtol[key]:idx for key,idx in opt.wtod.items() if key in opt.wtol}
        print('{} classes have the associated lemma word!'.format(len(lemma_det_dict)))

    if opt.eval_obj_grounding or opt.language_eval:
        for step in range(len(dataloader_val)):
            data = data_iter_val.next()
            if opt.vis_attn:
                seg_feat, iseq, gts_seq, num, proposals, bboxs, box_mask, seg_id, seg_show, seg_dim_info, region_feat, frm_mask, sample_idx, ppl_mask = data
            else:
                seg_feat, iseq, gts_seq, num, proposals, bboxs, box_mask, seg_id, region_feat, frm_mask, sample_idx, ppl_mask = data

            proposals = proposals[:,:max(int(max(num[:,1])),1),:]
            ppl_mask = ppl_mask[:,:max(int(max(num[:,1])),1)]
            region_feat = region_feat[:,:max(int(max(num[:,1])),1),:]

            segs_feat.resize_(seg_feat.size()).data.copy_(seg_feat)
            input_num.resize_(num.size()).data.copy_(num)
            input_ppls.resize_(proposals.size()).data.copy_(proposals)
            mask_ppls.resize_(ppl_mask.size()).data.copy_(ppl_mask)
            pnt_mask = torch.cat((mask_ppls.new(mask_ppls.size(0), 1).fill_(0), mask_ppls), dim=1) # pad 1 column from a legacy reason
            ppls_feat.resize_(region_feat.size()).data.copy_(region_feat)
            sample_idx = Variable(sample_idx.type(input_num.type()))

            eval_opt = {'sample_max':1, 'beam_size': opt.beam_size, 'inference_mode' : True}
            dummy = input_ppls.new(input_ppls.size(0)).byte().fill_(0)

            batch_size = input_ppls.size(0)

            seq, att2_weights, sim_mat = model(segs_feat, dummy, dummy, input_num, \
                                               input_ppls, dummy, dummy, ppls_feat, dummy, sample_idx, pnt_mask, 'sample', eval_opt)

            # save localization results on generated sentences
            if opt.eval_obj_grounding:
                assert opt.beam_size == 1, 'only support beam_size is 1'

                att2_ind = torch.max(att2_weights.view(batch_size, att2_weights.size(1), \
                    opt.num_sampled_frm, opt.num_prop_per_frm), dim=-1)[1]
                obj_bbox_att2 = torch.gather(input_ppls.view(-1, opt.num_sampled_frm, opt.num_prop_per_frm, 7) \
                    .permute(0, 2, 1, 3).contiguous(), 1, att2_ind.unsqueeze(-1).expand((batch_size, \
                    att2_ind.size(1), opt.num_sampled_frm, input_ppls.size(-1)))) # Bx20x10x7

                for i in range(seq.size(0)):
                    vid_id, seg_idx = seg_id[i].split('_segment_')
                    seg_idx = str(int(seg_idx))
                    tmp_result = {'clss':[], 'idx_in_sent':[], 'bbox_for_all_frames':[]}

                    for j in range(seq.size(1)):
                        if seq[i,j].item() != 0:
                            lemma = opt.wtol[opt.itow[str(seq[i,j].item())]]
                            if lemma in lemma_det_dict:
                                tmp_result['bbox_for_all_frames'].append(obj_bbox_att2[i, j, :, :4].tolist())
                                tmp_result['clss'].append(opt.itod[lemma_det_dict[lemma]])
                                tmp_result['idx_in_sent'].append(j) # redundant, for the sake of output format
                        else:
                            break
                    grd_output[vid_id][seg_idx] = tmp_result

            sents = utils.decode_sequence(dataset.itow, dataset.itod, dataset.ltow, dataset.itoc, \
                                          dataset.wtod, seq.data, opt.vocab_size, opt)

            for k, sent in enumerate(sents):
                vid_idx, seg_idx = seg_id[k].split('_segment_')
                seg_idx = int(seg_idx)

                predictions[vid_idx].append(
                    {'sentence':sent,
                    'timestamp':raw_caption_file[vid_idx]['timestamps'][seg_idx]})

                if num_show < 20:
                    print('segment %s: %s' %(seg_id[k], sent))
                    num_show += 1

                # visualization
                if opt.vis_attn:
                    assert(opt.beam_size == 1) # only support beam_size=1
                    att2_weights = F.softmax(att2_weights, dim=2)
                    # visualize some selected examples
                    if torch.sum(proposals[k]) != 0:
                        vis_infer(seg_show[k], seg_id[k], sent, \
                            att2_weights[k].cpu().data, proposals[k], num[k].long(), \
                            bboxs[k], sim_mat[k].cpu().data, seg_dim_info[k])

            if count % 2 == 0:
                print(count)
            count += 1

    lang_stats = defaultdict(float)
    if opt.language_eval:
        print('Total videos to be evaluated %d' %(len(predictions)))

        submission = 'results/'+'densecap-'+opt.val_split+'-'+opt.id+'.json'
        dense_cap_all = {'version':'VERSION 1.0', 'results':predictions,
                         'external_data':{'used':'true',
                          'details':'Visual Genome for Faster R-CNN pre-training'}}
        with open(submission, 'w') as f:
            json.dump(dense_cap_all, f)

        references = opt.densecap_references
        verbose = opt.densecap_verbose
        tious_lst = [0.3, 0.5, 0.7, 0.9]
        evaluator = ANETcaptions(ground_truth_filenames=references,
                             prediction_filename=submission,
                             tious=tious_lst,
                             max_proposals=1000,
                             verbose=verbose)
        evaluator.evaluate()

        for m,v in evaluator.scores.items():
            lang_stats[m] = np.mean(v)

        print('\nResults Summary (lang eval):')
        print('Printing language evaluation metrics...')
        for m, s in lang_stats.items():
            print('{}: {:.3f}'.format(m, s*100))
        print('\n')

    if opt.eval_obj_grounding:
        # write attention results to file
        attn_file = 'results/attn-gen-sent-results-'+opt.val_split+'-'+opt.id+'.json'
        with open(attn_file, 'w') as f:
            json.dump({'results':grd_output, 'eval_mode':'gen', 'external_data':{'used':True, 'details':'Object detector pre-trained on Visual Genome on object detection task.'}}, f)

        if not opt.test_mode:
            # offline eval
            evaluator = ANetGrdEval(reference_file=opt.grd_reference, submission_file=attn_file,
                                  split_file=opt.split_file, val_split=[opt.val_split],
                                  iou_thresh=0.5)

            print('\nResults Summary (generated sent):')
            print('Printing attention accuracy on generated sentences, per class and per sentence, respectively...')
            prec_all, recall_all, f1_all, prec_all_per_sent, rec_all_per_sent, f1_all_per_sent = evaluator.grd_eval(mode='all')
            prec_loc, recall_loc, f1_loc, prec_loc_per_sent, rec_loc_per_sent, f1_loc_per_sent = evaluator.grd_eval(mode='loc')
        else:
            print('*'*62)
            print('*  [WARNING] Grounding eval unavailable for the test set!\
    *\n*            Please submit your result files under directory *\
     \n*            results/ to the eval server!                    *')
            print('*'*62)

    if opt.att_model == 'topdown' and opt.eval_obj_grounding_gt:
        with torch.no_grad():
            box_accu_att, box_accu_grd, cls_accu = eval_grounding(opt) # eval grounding
    else:
        box_accu_att, box_accu_grd, cls_accu = 0, 0, 0

    if opt.enable_visdom:
        assert(opt.language_eval)
        if vis_window['score'] is None:
            vis_window['score'] = vis.line(
                X=np.tile(np.arange(epoch, epoch+1),
                          (7,1)).T,
                Y=np.column_stack((np.asarray(box_accu_att),
                                   np.asarray(box_accu_grd),
                                   np.asarray(cls_accu),
                                   np.asarray(lang_stats['Bleu_4']),
                                   np.asarray(lang_stats['METEOR']),
                                   np.asarray(lang_stats['CIDEr']),
                                   np.asarray(lang_stats['SPICE']))),
                opts=dict(title='Validation Score',
                          xlabel='Validation Epoch',
                          ylabel='Score',
                          legend=['BA (alpha)', 'BA (beta)', 'CLS Accu', 'Bleu_4', 'METEOR', 'CIDEr', 'SPICE'])
            )
        else:
            vis.line(
                X=np.tile(np.arange(epoch, epoch+1),
                          (7,1)).T,
                Y=np.column_stack((np.asarray(box_accu_att),
                                   np.asarray(box_accu_grd),
                                   np.asarray(cls_accu),
                                   np.asarray(lang_stats['Bleu_4']),
                                   np.asarray(lang_stats['METEOR']),
                                   np.asarray(lang_stats['CIDEr']),
                                   np.asarray(lang_stats['SPICE']))),
                opts=dict(title='Validation Score',
                          xlabel='Validation Epoch',
                          ylabel='Score',
                          legend=['BA (alpha)', 'BA (beta)', 'CLS Accu', 'Bleu_4', 'METEOR', 'CIDEr', 'SPICE']),
                win=vis_window['score'],
                update='append'
            )

    print('Saving the predictions')

    # Write validation result into summary
    val_result_history[iteration] = {'lang_stats': lang_stats, 'predictions': predictions}

    return lang_stats


if __name__ == '__main__':

    opt = opts.parse_opt()
    if opt.path_opt is not None:
        with open(opt.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle)
        utils.update_values(options_yaml, vars(opt))
    opt.test_mode = (opt.val_split in ['testing', 'hidden_test'])
    if opt.enable_BUTD:
        assert opt.att_input_mode == 'region', 'region attention only under the BUTD mode'

    # print(opt)
    cudnn.benchmark = True

    if opt.enable_visdom:
        import visdom
        vis = visdom.Visdom(server=opt.visdom_server, env=opt.id)
        vis_window={'iter': None, 'score':None}

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.seed)
    if opt.vis_attn:
        import cv2

    if opt.dataset == 'anet':
        from misc.dataloader_anet import DataLoader
    else:
        raise Exception('only support anet!')

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    # Data Loader
    dataset = DataLoader(opt, split=opt.train_split, seq_per_img=opt.seq_per_img)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=opt.num_workers)

    dataset_val = DataLoader(opt, split=opt.val_split, seq_per_img=opt.seq_per_img)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size,
                                            shuffle=False, num_workers=opt.num_workers)

    segs_feat = torch.FloatTensor(1)
    input_seqs = torch.LongTensor(1)
    input_ppls = torch.FloatTensor(1)
    mask_ppls = torch.ByteTensor(1)
    gt_bboxs = torch.FloatTensor(1)
    mask_bboxs = torch.ByteTensor(1)
    mask_frms = torch.ByteTensor(1)
    gt_seqs = torch.LongTensor(1)
    input_num = torch.LongTensor(1)
    ppls_feat = torch.FloatTensor(1)

    if opt.cuda:
        segs_feat = segs_feat.cuda()
        input_seqs = input_seqs.cuda()
        gt_seqs = gt_seqs.cuda()
        input_num = input_num.cuda()
        input_ppls = input_ppls.cuda()
        mask_ppls = mask_ppls.cuda()
        gt_bboxs = gt_bboxs.cuda()
        mask_bboxs = mask_bboxs.cuda()
        mask_frms = mask_frms.cuda()
        ppls_feat =  ppls_feat.cuda()

    segs_feat = Variable(segs_feat)
    input_seqs = Variable(input_seqs)
    gt_seqs = Variable(gt_seqs)
    input_num = Variable(input_num)
    input_ppls = Variable(input_ppls)
    mask_ppls = Variable(mask_ppls)
    gt_bboxs = Variable(gt_bboxs)
    mask_bboxs = Variable(mask_bboxs)
    mask_frms = Variable(mask_frms)
    ppls_feat = Variable(ppls_feat)

    # Build the Model
    opt.vocab_size = dataset.vocab_size
    opt.detect_size = dataset.detect_size
    opt.seq_length = opt.seq_length
    opt.glove_w = torch.from_numpy(dataset.glove_w).float()
    opt.glove_vg_cls = torch.from_numpy(dataset.glove_vg_cls).float()
    opt.glove_clss = torch.from_numpy(dataset.glove_clss).float()

    opt.wtoi = dataset.wtoi
    opt.itow = dataset.itow
    opt.itod = dataset.itod
    opt.ltow = dataset.ltow
    opt.itoc = dataset.itoc
    opt.wtol = dataset.wtol
    opt.wtod = dataset.wtod
    opt.vg_cls = dataset.vg_cls

    if opt.att_model == 'topdown':
        model = AttModel.TopDownModel(opt)
    elif opt.att_model == 'transformer':
        model = AttModel.TransformerModel(opt)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        if opt.load_best_score == 1:
            model_path = os.path.join(opt.start_from, 'model-best.pth')
            info_path = os.path.join(opt.start_from, 'infos_'+opt.id+'-best.pkl')
        else:
            model_path = os.path.join(opt.start_from, 'model.pth')
            info_path = os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')

        # open old infos and check if models are compatible
        with open(info_path, 'rb') as f:
            infos = pickle.load(f, encoding='latin1') # py2 pickle -> py3
            # infos = pickle.load(f)
            saved_model_opt = infos['opt']

        # opt.learning_rate = saved_model_opt.learning_rate
        print('Loading the model %s...' %(model_path))
        model.load_state_dict(torch.load(model_path))

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
                histories = pickle.load(f, encoding='latin1') # py2 pickle -> py3
                # histories = pickle.load(f)

    best_val_score = infos.get('best_val_score', None)
    iteration = infos.get('iter', 0)
    start_epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    if opt.mGPUs:
        model = nn.DataParallel(model)

    if opt.cuda:
        model.cuda()

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if ('ctx2pool_grd' in key) or ('vis_embed' in key):
                print('Finetune param: {}'.format(key))
                params += [{'params':[value], 'lr':opt.learning_rate*0.1, # finetune the fc7 layer
                    'weight_decay':opt.weight_decay, 'betas':(opt.optim_alpha, opt.optim_beta)}]
            else:
                params += [{'params':[value], 'lr':opt.learning_rate,
                    'weight_decay':opt.weight_decay, 'betas':(opt.optim_alpha, opt.optim_beta)}]

    print("Use %s as optmization method" %(opt.optim))
    if opt.optim == 'sgd':
        optimizer = optim.SGD(params, momentum=0.9)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(params)
    elif opt.optim == 'adamax':
    	optimizer = optim.Adamax(params)

    for epoch in range(start_epoch, opt.max_epochs):
        if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
            if (epoch - opt.learning_rate_decay_start) % opt.learning_rate_decay_every == 0:
                # decay the learning rate.
                utils.set_lr(optimizer, opt.learning_rate_decay_rate)
                opt.learning_rate  = opt.learning_rate * opt.learning_rate_decay_rate

        if not opt.inference_only:
            if opt.enable_visdom:
                train(epoch, opt, vis, vis_window)
            else:
                train(epoch, opt)

        if epoch % opt.val_every_epoch == 0:
            with torch.no_grad():
                if opt.enable_visdom:
                    lang_stats = eval(epoch, opt, vis, vis_window)
                else:
                    lang_stats = eval(epoch, opt)

            if opt.inference_only:
                break

            # Save model if is improving on validation result
            current_score = lang_stats['CIDEr']

            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
            if opt.mGPUs:
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            # optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            # torch.save(optimizer.state_dict(), optimizer_path)

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = dataset.itow

            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            histories['ss_prob_history'] = ss_prob_history
            with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                pickle.dump(infos, f)
            with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                pickle.dump(histories, f)

            if best_flag:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                if opt.mGPUs:
                    torch.save(model.module.state_dict(), checkpoint_path)
                else:
                    torch.save(model.state_dict(), checkpoint_path)

                print("model saved to {} with best cider score {:.3f}".format(checkpoint_path, best_val_score))
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                    pickle.dump(infos, f)
