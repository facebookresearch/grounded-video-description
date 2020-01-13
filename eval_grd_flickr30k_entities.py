# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Evaluation script for object localization

import json
import argparse
import torch
import itertools
import numpy as np
from collections import defaultdict
from misc.utils import bbox_overlaps_batch

from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm


class FlickrGrdEval(object):

    def __init__(self, reference_file=None, submission_file=None,
                 split_file=None, val_split=None, iou_thresh=0.5, verbose=False):

        if not reference_file:
            raise IOError('Please input a valid reference file!')
        if not submission_file:
            raise IOError('Please input a valid submission file!')

        self.iou_thresh = iou_thresh
        self.verbose = verbose
        self.val_split = val_split

        self.import_ref(reference_file, split_file)
        self.import_sub(submission_file)


    def import_ref(self, reference_file=None, split_file=None):

        with open(split_file) as f:
            split_dict = json.load(f)
        split = {}
        for s in self.val_split:
            split.update({i:i for i in split_dict[s]})

        with open(reference_file) as f:
            ref = json.load(f)['annotations']

        ref = [v for k,v in enumerate(ref) if str(v['image_id']) in split]
        self.ref = ref


    def import_sub(self, submission_file=None):

        with open(submission_file) as f:
            pred = json.load(f)['results']
        self.pred = pred


    def gt_grd_eval(self):

        ref = self.ref
        pred = self.pred
        print('Number of images in the reference: {}, number of images in the submission: {}'.format(len(ref), len(pred)))

        results = defaultdict(list)
        for lst_idx, anns in enumerate(ref):
            img = str(anns['image_id'])
            for num_sent, ann in enumerate(anns['captions']):
                ref_bbox_all = torch.Tensor(ann['process_bnd_box'])
                sent_idx = ann['process_idx'] # index of word in sentence to evaluate
                for idx in sent_idx:
                    sel_idx = [ind for ind, i in enumerate(ann['process_idx']) if idx == i]
                    assert(len(sel_idx) == 1)
                    ref_bbox = ref_bbox_all[sel_idx[0]] # select matched boxes
                    assert(ref_bbox.size(0) > 0)

                    class_name = ann['process_clss'][sel_idx[0]]
                    if img not in pred:
                        results[class_name].append(0) # image not grounded
                    elif len(pred[img]) != 5:
                        raise Exception('Each image must have five caption predictions!')
                    elif idx not in pred[img][num_sent]['idx_in_sent']:
                        results[class_name].append(0) # object not grounded
                    else:
                        pred_ind = pred[img][num_sent]['idx_in_sent'].index(idx)
                        pred_bbox = torch.Tensor(pred[img][num_sent]['bbox'][pred_ind])

                        overlap = bbox_overlaps_batch(pred_bbox.unsqueeze(0), \
                            ref_bbox.unsqueeze(0).unsqueeze(0))
                        results[class_name].append(1 if torch.max(overlap) > self.iou_thresh else 0)

        print('Number of groundable objects in this split: {}'.format(len(results)))
        grd_accu = np.mean([sum(hm)*1./len(hm) for i,hm in results.items()])

        print('-' * 80)
        print('The overall localization accuracy is {:.4f}'.format(grd_accu))
        print('-' * 80)
        if self.verbose:
            print('Object frequency and grounding accuracy per class (descending by object frequency):')
            accu_per_clss = {(i, sum(hm)*1./len(hm)):len(hm) for i,hm in results.items()}
            accu_per_clss = sorted(accu_per_clss.items(), key=lambda x:x[1], reverse=True)
            for accu in accu_per_clss:
                print('{} ({}): {:.4f}'.format(accu[0][0], accu[1], accu[0][1]))

        return grd_accu


    def grd_eval(self, mode='all'):

        if mode == 'all':
            print('Evaluating on all object words.')
        elif mode == 'loc':
            print('Evaluating only on correctly-predicted object words.')
        else:
            raise Exception('Invalid loc mode!')

        ref = self.ref
        pred = self.pred
        print('Number of images in the reference: {}, number of images in the submission: {}'.format(len(ref), len(pred)))

        nlp = StanfordCoreNLP('tools/stanford-corenlp-full-2018-02-27')
        props={'annotators': 'lemma','pipelineLanguage':'en', 'outputFormat':'json'}
        vocab_in_split = set()

        # precision
        prec = defaultdict(list)
        for lst_idx, anns in tqdm(enumerate(ref)):
            img = str(anns['image_id'])

            for num_sent, ann in enumerate(anns['captions']):
                if img not in pred:
                    continue # do not penalize if sentence not annotated
                assert(len(pred[img]) == 1)

                ref_bbox_all = torch.Tensor(ann['process_bnd_box'])

                idx_in_sent = {}
                for box_idx, cls in enumerate(ann['process_clss']):
                    vocab_in_split.update(set([cls]))
                    idx_in_sent[cls] = idx_in_sent.get(cls, []) + [ann['process_idx'][box_idx]]

                sent_idx = ann['process_idx'] # index of gt object words
                exclude_obj = {json.loads(nlp.annotate(token, properties=props) \
                    )['sentences'][0]['tokens'][0]['lemma']:1 for token_idx, token in enumerate(ann['tokens'] \
                    ) if (token_idx not in sent_idx and token != '')}

                for pred_idx, class_name in enumerate(pred[img][0]['clss']):
                    if class_name in idx_in_sent:
                        gt_idx = min(idx_in_sent[class_name]) # always consider the first match...
                        sel_idx = [idx for idx, i in enumerate(ann['process_idx']) if gt_idx == i]
                        assert(len(sel_idx) == 1)
                        ref_bbox = ref_bbox_all[sel_idx[0]] # select matched boxes
                        assert(ref_bbox.size(0) > 0)

                        pred_bbox = torch.Tensor(pred[img][0]['bbox'][pred_idx])

                        overlap = bbox_overlaps_batch(pred_bbox.unsqueeze(0), \
                            ref_bbox.unsqueeze(0).unsqueeze(0))
                        prec[class_name].append(1 if torch.max(overlap) > self.iou_thresh else 0)
                    elif json.loads(nlp.annotate(class_name, properties=props))['sentences'][0]['tokens'][0]['lemma'] in exclude_obj:
                        pass # do not penalize if gt object word not annotated (missed)
                    else:
                        if mode == 'all':
                            prec[class_name].append(0) # hallucinated object

        nlp.close()

        # recall
        recall = defaultdict(list)
        for lst_idx, anns in enumerate(ref):
            img = str(anns['image_id'])
            for num_sent, ann in enumerate(anns['captions']):
                ref_bbox_all = torch.Tensor(ann['process_bnd_box'])
                sent_idx = ann['process_idx'] # index of gt object words

                for gt_idx in sent_idx:
                    sel_idx = [idx for idx, i in enumerate(ann['process_idx']) if gt_idx == i]
                    assert(len(sel_idx) == 1)
                    ref_bbox = ref_bbox_all[sel_idx[0]] # select matched boxes
                    assert(ref_bbox.size(0) > 0)

                    class_name = ann['process_clss'][sel_idx[0]]
                    if img not in pred:
                        recall[class_name].append(0) # image not grounded
                    elif class_name in pred[img][0]['clss']:
                        pred_idx = pred[img][0]['clss'].index(class_name) # always consider the first match...
                        pred_bbox = torch.Tensor(pred[img][0]['bbox'][pred_idx])

                        overlap = bbox_overlaps_batch(pred_bbox.unsqueeze(0), \
                            ref_bbox.unsqueeze(0).unsqueeze(0))
                        recall[class_name].append(1 if torch.max(overlap) > self.iou_thresh else 0)
                    else:
                        if mode == 'all':
                            recall[class_name].append(0) # object not grounded

        num_vocab = len(vocab_in_split)
        print('Number of groundable objects in this split: {}'.format(num_vocab))
        print('Number of objects in prec and recall: {}, {}'.format(len(prec), len(recall)))
        prec_accu = np.sum([sum(hm)*1./len(hm) for i,hm in prec.items()])*1./num_vocab
        recall_accu = np.sum([sum(hm)*1./len(hm) for i,hm in recall.items()])*1./num_vocab
        f1 = 2. * prec_accu * recall_accu / (prec_accu + recall_accu)

        print('-' * 80)
        print('The overall precision_{0} / recall_{0} / F1_{0} are {1:.4f} / {2:.4f} / {3:.4f}'.format(mode, prec_accu, recall_accu, f1))
        print('-' * 80)
        if self.verbose:
            print('Object frequency and grounding accuracy per class (descending by object frequency):')
            accu_per_clss = {}
            for i in vocab_in_split:
                prec_clss = sum(prec[i])*1./len(prec[i]) if i in prec else 0
                recall_clss = sum(recall[i])*1./len(recall[i]) if i in recall else 0
                accu_per_clss[(i, prec_clss, recall_clss)] = (len(prec[i]), len(recall[i]))
            accu_per_clss = sorted(accu_per_clss.items(), key=lambda x:x[1][1], reverse=True)
            for accu in accu_per_clss:
                print('{} ({} / {}): {:.4f} / {:.4f}'.format(accu[0][0], accu[1][0], accu[1][1], accu[0][1], accu[0][2]))

        return prec_accu, recall_accu, f1


def main(args):

    grd_evaluator = FlickrGrdEval(reference_file=args.reference, submission_file=args.submission,
                           split_file=args.split_file, val_split=args.split,
                           iou_thresh=args.iou_thresh, verbose=args.verbose)
    if args.eval_mode == 'GT':
        print('Assuming the input boxes are based upon GT sentences.')
        grd_evaluator.gt_grd_eval()
    elif args.eval_mode == 'gen':
        print('Assuming the input boxes are based upon generated sentences.')
        grd_evaluator.grd_eval(mode=args.loc_mode)
    else:
        raise Exception('Invalid eval mode!')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='ActivityNet-Entities object grounding evaluation script.')
    parser.add_argument('-s', '--submission', type=str, default='', help='submission grounding result file')
    parser.add_argument('-r', '--reference', type=str, default='data/flickr30k/flickr30k_cleaned_class.json', help='reference file')
    parser.add_argument('--split_file', type=str, default='data/flickr30k/split_ids_flickr30k_entities.json', help='path to the split file')
    parser.add_argument('--split', type=str, nargs='+', default=['val'], help='which split(s) to evaluate')

    parser.add_argument('--eval_mode', type=str, default='GT',
        help='GT | gen, indicating whether the input is on GT sentences or generated sentences')
    parser.add_argument('--loc_mode', type=str, default='all',
        help='all | loc, when the input is on generate sentences, whether consider language error or not')

    parser.add_argument('--iou_thresh', type=float, default=0.5, help='the iou threshold for grounding correctness')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
