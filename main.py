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

import opts
from misc import utils, AttModel
from collections import defaultdict

import torchvision.transforms as transforms
import pdb

from eval_grd_flickr30k_entities import FlickrGrdEval


# visualization over generated sentences
def vis_infer(img_show, img_id, caption, att2_weights, proposals, sim_mat):
    cap = caption.split()
    output = []
    top_k_prop = 1 # plot the top 1 proposal only

    sim_mat_val, sim_mat_ind = torch.max(sim_mat, dim=0)

    for j in range(len(cap)):
            max_att2_weight, top_k_alpha_idx = torch.max(att2_weights[j], dim=0)
            img = copy.deepcopy(img_show.numpy())
            img_text = np.ones((67, img.shape[1], 3))*255
            cv2.putText(img_text, '%s' % (cap[j]), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 0, 0), thickness=3)

            # draw the proposal box and text
            idx = top_k_alpha_idx
            bbox = tuple(int(np.round(x)) for x in proposals[idx, :4])
            class_name = opt.itod.get(sim_mat_ind[idx].item(), '__background__')
            cv2.rectangle(img, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(img, '%s: (%.2f)' % (class_name, sim_mat_val[idx]),
                       (bbox[0], bbox[1] + 25), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), thickness=2)
            output.append(np.concatenate([img_text, img], axis=0))

    output = np.concatenate(output, axis=1)
    if not os.path.isdir('./vis/'+opt.id):
        os.mkdir('./vis/'+opt.id)
    # print('Visualize image {} and the generated sentence!'.format(img_id))
    cv2.imwrite('./vis/'+opt.id+'/'+str(img_id)+'_generated_sent.jpg', output[:,:,::-1])


# compute localization (attention/grounding) accuracy over GT sentences
def eval_grounding(opt):
    model.eval()

    data_iter = iter(dataloader_val)
    cls_pred_lst = []
    cls_accu_score = defaultdict(list)
    att2_output = defaultdict(list)
    grd_output = defaultdict(list)
    vocab_in_split = set()

    for step in range(len(dataloader_val)):
        data = data_iter.next()
        if opt.vis_attn:
            img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id, img_show, region_feat = data
        else:
            img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id, region_feat = data
        proposals = proposals[:,:max(int(max(num[:,1])),1),:]
        bboxs = bboxs[:,:int(max(num[:,2])),:]
        region_feat = region_feat[:,:max(int(max(num[:,1])),1),:]

        input_imgs.resize_(img.size()).data.copy_(img)
        input_seqs.resize_(iseq.size()).data.copy_(iseq)
        gt_seqs.resize_(gts_seq.size()).data.copy_(gts_seq)
        input_num.resize_(num.size()).data.copy_(num)
        input_ppls.resize_(proposals.size()).data.copy_(proposals)
        gt_bboxs.resize_(bboxs.size()).data.copy_(bboxs) # for region cls eval only
        ppls_feat.resize_(region_feat.size()).data.copy_(region_feat)

        dummy = input_ppls.new(input_ppls.size(0)).byte().fill_(0)

        # cls_pred_hm_lst contains a list of tuples (clss_ind, hit/1 or miss/0)
        cls_pred_hm_lst, att2_ind, grd_ind = model(input_imgs, input_seqs, gt_seqs, input_num,
            input_ppls, gt_bboxs, dummy, ppls_feat, 'GRD')

        # save attention/grounding results on GT sentences
        att2_ind = att2_ind.view(-1, opt.seq_per_img, att2_ind.size(1))
        grd_ind = grd_ind.view(-1, opt.seq_per_img, grd_ind.size(1))

        # resize proposals back
        input_ppls[:, :, torch.LongTensor([0, 2])] *= input_num[:,3].float().view(-1, 1, 1)/opt.image_crop_size
        input_ppls[:, :, torch.LongTensor([1, 3])] *= input_num[:,4].float().view(-1, 1, 1)/opt.image_crop_size

        obj_mask = (input_seqs[:,:,1:,0] > opt.vocab_size) # Bx5x20
        obj_bbox_att2 = torch.gather(input_ppls.unsqueeze(1).expand(input_ppls.size(0), opt.seq_per_img,
            input_ppls.size(1), input_ppls.size(2)), 2, att2_ind.unsqueeze(-1).expand((att2_ind.size(0),
            att2_ind.size(1), att2_ind.size(2), input_ppls.size(-1)))) # Bx5x20x6
        obj_bbox_grd = torch.gather(input_ppls.unsqueeze(1).expand(input_ppls.size(0), opt.seq_per_img,
            input_ppls.size(1), input_ppls.size(2)), 2, grd_ind.unsqueeze(-1).expand((grd_ind.size(0),
            grd_ind.size(1), grd_ind.size(2), input_ppls.size(-1)))) # Bx5x20x6

        for i in range(obj_mask.size(0)):
            for num_sent in range(obj_mask.size(1)):
                tmp_result_grd = {'clss':[], 'idx_in_sent':[], 'bbox':[]}
                tmp_result_att2 = {'clss':[], 'idx_in_sent':[], 'bbox':[]}
                for j in range(obj_mask.size(2)):
                    if obj_mask[i, num_sent, j]:
                        cls_name = opt.itod[input_seqs[i, num_sent, j+1, 0].item()-opt.vocab_size]
                        vocab_in_split.update([cls_name])
                        tmp_result_att2['clss'].append(cls_name)
                        tmp_result_att2['idx_in_sent'].append(j)
                        tmp_result_att2['bbox'].append(obj_bbox_att2[i, num_sent, j, :4].tolist())
                        tmp_result_grd['clss'].append(cls_name)
                        tmp_result_grd['idx_in_sent'].append(j)
                        tmp_result_grd['bbox'].append(obj_bbox_grd[i, num_sent, j, :4].tolist())
                att2_output[img_id[i].item()].append(tmp_result_att2)
                grd_output[img_id[i].item()].append(tmp_result_grd)

        cls_pred_lst.append(cls_pred_hm_lst)

    # write results to file
    attn_file = 'results/attn-gt-sent-results-'+opt.val_split+'-'+opt.id+'.json'
    with open(attn_file, 'w') as f:
        json.dump({'results':att2_output, 'eval_mode':'GT', 'external_data':{'used':True, 'details':'Object detector pre-trained on Visual Genome on object detection task.'}}, f)
    grd_file = 'results/grd-gt-sent-results-'+opt.val_split+'-'+opt.id+'.json'
    with open(grd_file, 'w') as f:
        json.dump({'results':grd_output, 'eval_mode':'GT', 'external_data':{'used':True, 'details':'Object detector pre-trained on Visual Genome on object detection task.'}}, f)

    cls_pred_lst = torch.cat(cls_pred_lst, dim=0)
    for i in range(cls_pred_lst.size(0)):
        cls_accu_score[cls_pred_lst[i,0].long().item()].append(cls_pred_lst[i,1].item())

    print('Total number of object classes in the split: {}. {} have classification results.'.format(len(vocab_in_split), len(cls_accu_score)))
    cls_accu = np.sum([sum(hm)*1./len(hm) for i,hm in cls_accu_score.items()])*1./len(vocab_in_split)

    # offline eval
    evaluator = FlickrGrdEval(reference_file=opt.grd_reference, submission_file=attn_file,
                              split_file=opt.split_file, val_split=[opt.val_split],
                              iou_thresh=0.5)

    attn_accu = evaluator.gt_grd_eval()
    evaluator.import_sub(grd_file)
    grd_accu = evaluator.gt_grd_eval()

    return attn_accu, grd_accu, cls_accu


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
        img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id, region_feat = data
        proposals = proposals[:,:max(int(max(num[:,1])),1),:]
        bboxs = bboxs[:,:int(max(num[:,2])),:]
        box_mask = box_mask[:,:,:max(int(max(num[:,2])),1),:]
        region_feat = region_feat[:,:max(int(max(num[:,1])),1),:]

        input_imgs.resize_(img.size()).data.copy_(img)
        input_seqs.resize_(iseq.size()).data.copy_(iseq)
        gt_seqs.resize_(gts_seq.size()).data.copy_(gts_seq)
        input_num.resize_(num.size()).data.copy_(num)
        input_ppls.resize_(proposals.size()).data.copy_(proposals)
        gt_bboxs.resize_(bboxs.size()).data.copy_(bboxs)
        mask_bboxs.resize_(box_mask.size()).data.copy_(box_mask)
        ppls_feat.resize_(region_feat.size()).data.copy_(region_feat)

        loss = 0
        lm_loss, att2_loss, ground_loss, cls_loss = model(input_imgs, input_seqs, gt_seqs, input_num, input_ppls, gt_bboxs, mask_bboxs, ppls_feat, 'MLE')

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

            print("step {}/{} (epoch {}), lm_loss = {:.3f}, att2_loss = {:.3f}, ground_loss = {:.3f}, cls_los = {:.3f}, lr = {:.5f}, time/batch = {:.3f}" \
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
    predictions = []
    count = 0

    if opt.eval_obj_grounding:
        grd_output = defaultdict(list)

        lemma_det_dict = {opt.wtol[key]:idx for key,idx in opt.wtod.items() if key in opt.wtol}

        print('{} classes have the associated lemma word!'.format(len(lemma_det_dict)))

    if opt.eval_obj_grounding or opt.language_eval:
        for step in range(len(dataloader_val)):
            data = data_iter_val.next()
            if opt.vis_attn:
                img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id, img_show, region_feat = data
            else:
                img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id, region_feat = data

            proposals = proposals[:,:max(int(max(num[:,1])),1),:]
            region_feat = region_feat[:,:max(int(max(num[:,1])),1),:]

            input_imgs.resize_(img.size()).data.copy_(img)
            input_num.resize_(num.size()).data.copy_(num)
            input_ppls.resize_(proposals.size()).data.copy_(proposals)
            ppls_feat.resize_(region_feat.size()).data.copy_(region_feat)

            eval_opt = {'sample_max':1, 'beam_size': opt.beam_size, 'inference_mode' : True, 'tag_size' : opt.cbs_tag_size}
            dummy = input_ppls.new(input_imgs.size(0)).fill_(0)
            seq, att2_weights, sim_mat = model(input_imgs, dummy, dummy, input_num, \
                                               input_ppls, dummy, dummy, ppls_feat, 'sample', eval_opt)

            att2_weights_clone = att2_weights.clone()

            # save localization results on generated sentences
            if opt.eval_obj_grounding:
                assert opt.beam_size == 1, 'only support beam_size is 1'

                att2_ind = torch.max(att2_weights, dim=2)[1]

                # resize proposals back
                input_ppls[:, :, torch.LongTensor([0, 2])] *= input_num[:,3].float().view(-1, 1, 1)/opt.image_crop_size
                input_ppls[:, :, torch.LongTensor([1, 3])] *= input_num[:,4].float().view(-1, 1, 1)/opt.image_crop_size

                for i in range(seq.size(0)):
                    tmp_result = {'clss':[], 'idx_in_sent':[], 'bbox':[]}
                    num_sent = 0 # does not really matter which reference to use
                    for j in range(seq.size(1)):
                        if seq[i,j].item() != 0:
                            lemma = opt.wtol[opt.itow[str(seq[i,j].item())]]
                            if lemma in lemma_det_dict:
                                tmp_result['bbox'].append(input_ppls[i, att2_ind[i, j], :4].tolist())
                                tmp_result['clss'].append(opt.itod[lemma_det_dict[lemma]])
                                tmp_result['idx_in_sent'].append(j) # redundant, for the sake of output format
                        else:
                            break

                    grd_output[img_id[i].item()].append(tmp_result)

            sents = utils.decode_sequence(dataset.itow, dataset.itod, dataset.ltow, dataset.itoc, \
                                          dataset.wtod, seq.data, opt.vocab_size, opt)

            for k, sent in enumerate(sents):
                entry = {'image_id': img_id[k].item(), 'caption': sent}
                predictions.append(entry)
                if num_show < 20:
                    print('image %s: %s' %(entry['image_id'], entry['caption']))
                    num_show += 1

                # visualize the caption and region
                if opt.vis_attn:
                    if torch.sum(proposals[k]) != 0:
                        vis_infer(img_show[k], entry['image_id'], entry['caption'], att2_weights[k].cpu().data, proposals[k].data, sim_mat[k].cpu().data)
                        print('GT sent: {} \nattn prec (obj): {:.3f} ({}), recall (obj): {:.3f} ({})' \
                                .format('UNK', np.mean(ba_per_sent_prec[img_id[k].item()]), len(ba_per_sent_prec[img_id[k].item()]),
                                np.mean(ba_per_sent_recall[img_id[k].item()]), len(ba_per_sent_recall[img_id[k].item()])))
                        print('*'*80)


            if count % 2 == 0:
                print(count)
            count += 1

    lang_stats = None
    if opt.language_eval:
        print('Total image to be evaluated %d' %(len(predictions)))
        lang_stats = utils.language_eval(opt.dataset, predictions, opt.id, opt.val_split, opt)

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

        # offline eval
        evaluator = FlickrGrdEval(reference_file=opt.grd_reference, submission_file=attn_file,
                              split_file=opt.split_file, val_split=[opt.val_split],
                              iou_thresh=0.5)

        print('\nResults Summary (generated sent):')
        print('Printing attention accuracy on generated sentences...')
        prec_all, recall_all, f1_all = evaluator.grd_eval(mode='all')
        prec_loc, recall_loc, f1_loc = evaluator.grd_eval(mode='loc')
        print('\n')

    if opt.eval_obj_grounding_gt:
        box_accu_att, box_accu_grd, cls_accu = eval_grounding(opt)
        print('\nResults Summary (GT sent):')
        print('The averaged attention / grounding box accuracy across all classes is: {:.4f} / {:.4f}'.format(box_accu_att, box_accu_grd))
        print('The averaged classification accuracy across all classes is: {:.4f}\n'.format(cls_accu))
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
    if opt.enable_BUTD:
        assert opt.att_input_mode == 'region', 'region attention only under the BUTD mode'

    # print(opt)
    cudnn.benchmark = True

    if opt.enable_visdom:
        import visdom
        vis = visdom.Visdom()
        vis_window={'iter': None, 'score':None}

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.seed)
    if opt.vis_attn:
        import cv2

    if opt.dataset == 'flickr30k':
        from misc.dataloader_flickr30k import DataLoader
    else:
        raise Exception('only support flickr30k!')

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    dataset = DataLoader(opt, split=opt.train_split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                            shuffle=False, num_workers=opt.num_workers)

    dataset_val = DataLoader(opt, split=opt.val_split)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size,
                                            shuffle=False, num_workers=opt.num_workers)

    input_imgs = torch.FloatTensor(1)
    input_seqs = torch.LongTensor(1)
    input_ppls = torch.FloatTensor(1)
    gt_bboxs = torch.FloatTensor(1)
    mask_bboxs = torch.ByteTensor(1)
    gt_seqs = torch.LongTensor(1)
    input_num = torch.LongTensor(1)
    ppls_feat = torch.FloatTensor(1)

    if opt.cuda:
        input_imgs = input_imgs.cuda()
        input_seqs = input_seqs.cuda()
        gt_seqs = gt_seqs.cuda()
        input_num = input_num.cuda()
        input_ppls = input_ppls.cuda()
        gt_bboxs = gt_bboxs.cuda()
        mask_bboxs = mask_bboxs.cuda()
        ppls_feat =  ppls_feat.cuda()

    input_imgs = Variable(input_imgs)
    input_seqs = Variable(input_seqs)
    gt_seqs = Variable(gt_seqs)
    input_num = Variable(input_num)
    input_ppls = Variable(input_ppls)
    gt_bboxs = Variable(gt_bboxs)
    mask_bboxs = Variable(mask_bboxs)
    ppls_feat = Variable(ppls_feat)

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
    opt.vg_cls = dataset.vg_cls
    opt.wtol = dataset.wtol
    opt.wtod = dataset.wtod

    if not opt.finetune_cnn: opt.fixed_block = 4 # if not finetune, fix all cnn block

    if opt.att_model == 'topdown':
        model = AttModel.TopDownModel(opt)
    else:
        raise Exception('only support topdown!')

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
            if 'cnn' in key:
                params += [{'params':[value], 'lr':opt.cnn_learning_rate,
                        'weight_decay':opt.cnn_weight_decay, 'betas':(opt.cnn_optim_alpha, opt.cnn_optim_beta)}]
            elif ('ctx2pool_grd' in key) or ('vis_embed' in key):
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
