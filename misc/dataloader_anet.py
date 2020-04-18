# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
from torchvision.datasets.folder import default_loader
import torch
import torch.utils.data as data
import copy
from PIL import Image
import torchvision.transforms as transforms
import torchtext.vocab as vocab # use this to load glove vector
from collections import defaultdict

class DataLoader(data.Dataset):
    def __init__(self, opt, split='training', seq_per_img=5):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.seq_length = opt.seq_length
        self.split = split
        self.seq_per_img = seq_per_img
        self.att_feat_size = opt.att_feat_size
        self.vis_attn = opt.vis_attn
        self.feature_root = opt.feature_root
        self.seg_feature_root = opt.seg_feature_root
        self.num_sampled_frm = opt.num_sampled_frm
        self.num_prop_per_frm = opt.num_prop_per_frm
        self.exclude_bgd_det = opt.exclude_bgd_det
        self.prop_thresh = opt.prop_thresh
        self.t_attn_size = opt.t_attn_size
        self.test_mode = opt.test_mode
        self.max_gt_box = 100
        self.max_proposal = self.num_sampled_frm * self.num_prop_per_frm
        self.glove = vocab.GloVe(name='6B', dim=300)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_dic)
        self.info = json.load(open(self.opt.input_dic))
        self.itow = self.info['ix_to_word']
        self.wtoi = {w:i for i,w in self.itow.items()}
        self.wtod = {w:i+1 for w,i in self.info['wtod'].items()} # word to detection
        self.dtoi = self.wtod # detection to index
        self.itod = {i:w for w,i in self.dtoi.items()}
        self.wtol = self.info['wtol']
        self.ltow = {l:w for w,l in self.wtol.items()}
        self.vocab_size = len(self.itow) + 1 # since it start from 1
        print('vocab size is ', self.vocab_size)
        self.itoc = self.itod

        # get the glove vector for the vg detection cls
        obj_cls_file = 'data/vg_object_vocab.txt' # From Peter's repo
        with open(obj_cls_file) as f:
            data = f.readlines()
            classes = ['__background__']
            classes.extend([i.strip() for i in data])

        # for VG classes
        self.vg_cls = classes
        self.glove_vg_cls = np.zeros((len(classes), 300))
        for i, w in enumerate(classes):
                split_word = w.replace(',', ' ').split(' ')
                vector = []
                for word in split_word:
                    if word in self.glove.stoi:
                        vector.append(self.glove.vectors[self.glove.stoi[word]].numpy())
                    else: # use a random vector instead
                        vector.append(2*np.random.rand(300) - 1)

                avg_vector = np.zeros((300))
                for v in vector:
                    avg_vector += v

                self.glove_vg_cls[i] = avg_vector/len(vector)

        # open the caption json file
        print('DataLoader loading input file: ', opt.input_json)
        self.caption_file = json.load(open(self.opt.input_json))

        # open the caption json file with segment boundaries
        print('DataLoader loading grounding file: ', opt.grd_reference)
        self.timestamp_file = json.load(open(opt.grd_reference))

        # open the detection json file.
        print('DataLoader loading proposal file: ', opt.proposal_h5)
        h5_proposal_file = h5py.File(self.opt.proposal_h5, 'r', driver='core')
        self.num_proposals = h5_proposal_file['dets_num'][:]
        self.label_proposals = h5_proposal_file['dets_labels'][:]
        h5_proposal_file.close()

        # category id to labels. +1 becuase 0 is the background label.
        self.glove_clss = np.zeros((len(self.itod)+1, 300))
        self.glove_clss[0] = 2*np.random.rand(300) - 1 # background
        for i, word in enumerate(self.itod.values()):
        	if word in self.glove.stoi:
        		vector = self.glove.vectors[self.glove.stoi[word]]
        	else: # use a random vector instead
        		vector = 2*np.random.rand(300) - 1
        	self.glove_clss[i+1] = vector        	

        self.glove_w = np.zeros((len(self.wtoi)+1, 300))
        for i, word in enumerate(self.wtoi.keys()):
            vector = np.zeros((300))
            count = 0
            for w in word.split(' '):
                count += 1
                if w in self.glove.stoi:
                    glove_vector = self.glove.vectors[self.glove.stoi[w]]
                    vector += glove_vector.numpy()
                else: # use a random vector instead
                    random_vector = 2*np.random.rand(300) - 1
                    vector += random_vector
            self.glove_w[i+1] = vector / count            

        self.detect_size = len(self.itod)

        # separate out indexes for each of the provided splits
        self.split_ix = []
        self.num_seg_per_vid = defaultdict(list)
        for ix in range(len(self.info['videos'])):
            seg = self.info['videos'][ix]
            seg_id = seg['id']
            vid_id, seg_idx = seg_id.split('_segment_')
            self.num_seg_per_vid[vid_id].append(int(seg_idx))
            if seg['split'] == split:
                # all the feature files must exist
                if os.path.isfile(os.path.join(self.feature_root, seg_id+'.npy')) and \
                    os.path.isfile(os.path.join(self.seg_feature_root, vid_id[2:]+'_bn.npy')):
                    if opt.vis_attn:
                        if random.random() < 0.001: # randomly sample 0.1% segments to visualize
                            self.split_ix.append(ix)
                    else:
                        self.split_ix.append(ix)
        print('assigned %d segments to split %s' %(len(self.split_ix), split))

    def get_det_word(self, gt_bboxs, caption, bbox_ann):
        
        # get the present category.
        pcats = []
        for i in range(gt_bboxs.shape[0]):
            pcats.append(gt_bboxs[i,6])
        # get the orginial form of the caption.
        indicator = []

        indicator.append([(0, 0, 0)]*len(caption)) # category class, binary class, fine-grain class.
        for i, bbox in enumerate(bbox_ann):
            # if the bbox_idx is not filtered out.
            if bbox['bbox_idx'] in pcats:
                w_idx = bbox['idx']
                ng = bbox['clss']
                bn = (ng != caption[w_idx]) + 1
                fg = bbox['label']
                indicator[0][w_idx] = (self.wtod[bbox['clss']], bn, fg)

        return indicator

    def get_frm_mask(self, proposals, gt_bboxs):
        # proposals: num_pps
        # gt_bboxs: num_box
        num_pps = proposals.shape[0]
        num_box = gt_bboxs.shape[0]
        return (np.tile(proposals.reshape(-1,1), (1,num_box)) != np.tile(gt_bboxs, (num_pps,1)))

    def __getitem__(self, index):

        ix = self.split_ix[index]

        seg_id = self.info['videos'][ix]['id']
        vid_id_ix, seg_id_ix = seg_id.split('_segment_')
        seg_id_ix = str(int(seg_id_ix))

        # load the proposal file
        num_proposal = int(self.num_proposals[ix])
        proposals = copy.deepcopy(self.label_proposals[ix])
        proposals = proposals[:num_proposal,:]

        # no need to resize proposal nor GT box since they are all based on images with 720px in width)
        region_feature = np.load(os.path.join(self.feature_root, seg_id+'.npy'))
        region_feature = region_feature.reshape(-1, region_feature.shape[2]).copy()
        assert(num_proposal == region_feature.shape[0])

        # proposal mask to filter out low-confidence proposals or backgrounds
        pnt_mask = (proposals[:, 6] <= self.prop_thresh)
        if self.exclude_bgd_det:
            pnt_mask |= (proposals[:, 5] == 0)

        # load the frame-wise segment feature
        seg_rgb_feature = np.load(os.path.join(self.seg_feature_root, vid_id_ix[2:]+'_resnet.npy'))
        seg_motion_feature = np.load(os.path.join(self.seg_feature_root, vid_id_ix[2:]+'_bn.npy'))
        seg_feature_raw = np.concatenate((seg_rgb_feature, seg_motion_feature), axis=1)

        # not accurate, with minor misalignments
        timestamps = self.timestamp_file['annotations'][vid_id_ix]['segments'][str(int(seg_id_ix))]['timestamps']
        dur = self.timestamp_file['annotations'][vid_id_ix]['duration']
        num_frm = seg_feature_raw.shape[0]
        sample_idx = np.array([np.round(num_frm*timestamps[0]*1./dur), np.round(num_frm*timestamps[1]*1./dur)])
        sample_idx = np.clip(np.round(sample_idx), 0, self.t_attn_size).astype(int)
        seg_feature = np.zeros((self.t_attn_size, seg_feature_raw.shape[1]))
        seg_feature[:min(self.t_attn_size, num_frm)] = seg_feature_raw[:self.t_attn_size]

        captions = [copy.deepcopy(self.caption_file[vid_id_ix]['segments'][seg_id_ix])] # one per segment
        assert len(captions) == 1,  'Only support one caption per segment for now!'

        bbox_ann = []
        bbox_idx = 0
        for caption in captions:
            for i, clss in enumerate(caption['clss']):
                for j, cls in enumerate(clss): # one box might have multiple labels
                    # we don't care about the boxes outside the length limit.
                    # after all our goal is referring, not detection
                    if caption['idx'][i][j] < self.seq_length:
                        if self.test_mode:
                            # dummy bbox and frm_idx for the hidden testing split
                            bbox_ann.append({'bbox':[0, 0, 0, 0], 'label': self.dtoi[cls], 'clss': cls,
                                'bbox_idx':bbox_idx, 'idx':caption['idx'][i][j], 'frm_idx':-1})
                        else:
                            bbox_ann.append({'bbox':caption['bbox'][i], 'label': self.dtoi[cls], 'clss': cls,
                                'bbox_idx':bbox_idx, 'idx':caption['idx'][i][j], 'frm_idx':caption['frm_idx'][i]})

                        bbox_idx += 1

        # (optional) sort the box based on idx
        bbox_ann = sorted(bbox_ann, key=lambda x:x['idx'])

        gt_bboxs = np.zeros((len(bbox_ann), 8))
        for i, bbox in enumerate(bbox_ann):
            gt_bboxs[i, :4] = bbox['bbox']
            gt_bboxs[i, 4] = bbox['frm_idx']
            gt_bboxs[i, 5] = bbox['label']
            gt_bboxs[i, 6] = bbox['bbox_idx']
            gt_bboxs[i, 7] = bbox['idx']

        if not self.test_mode: # skip this in test mode
            gt_x = (gt_bboxs[:,2]-gt_bboxs[:,0]+1)
            gt_y = (gt_bboxs[:,3]-gt_bboxs[:,1]+1)
            gt_area_nonzero = (((gt_x != 1) & (gt_y != 1)))
            gt_bboxs = gt_bboxs[gt_area_nonzero]

        # given the bbox_ann, and caption, this function determine which word belongs to the detection.
        det_indicator = self.get_det_word(gt_bboxs, captions[0]['caption'], bbox_ann)
        # fetch the captions
        ncap = len(captions) # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        # convert caption into sequence label.
        cap_seq = np.zeros([ncap, self.seq_length, 5])
        for i, caption in enumerate(captions):
            j = 0
            while j < len(caption['caption']) and j < self.seq_length:
                is_det = False
                if det_indicator[i][j][0] != 0:
                    cap_seq[i,j,0] = det_indicator[i][j][0] + self.vocab_size
                    cap_seq[i,j,1] = det_indicator[i][j][1]
                    cap_seq[i,j,2] = det_indicator[i][j][2]
                    cap_seq[i,j,3] = self.wtoi[caption['caption'][j]]
                    cap_seq[i,j,4] = self.wtoi[caption['caption'][j]]
                else:
                    cap_seq[i,j,0] = self.wtoi[caption['caption'][j]]
                    cap_seq[i,j,4] = self.wtoi[caption['caption'][j]]
                j += 1

        # get the mask of the ground truth bounding box. The data shape is 
        # num_caption x num_box x num_seq
        box_mask = np.ones((len(captions), gt_bboxs.shape[0], self.seq_length))
        for i in range(gt_bboxs.shape[0]):
            box_mask[0, i, int(gt_bboxs[i][7])] = 0

        gt_bboxs = gt_bboxs[:,:6]

        # get the batch version of the seq and box_mask.
        if ncap < self.seq_per_img:
            seq_batch = np.zeros([self.seq_per_img, self.seq_length, 4])
            mask_batch = np.zeros([self.seq_per_img, gt_bboxs.shape[0], self.seq_length])
            # we need to subsample (with replacement)
            for q in range(self.seq_per_img):
                ixl = random.randint(0,ncap)
                seq_batch[q,:] = cap_seq[ixl,:,:4]
                mask_batch[q,:] = box_mask[ixl]
        else:
            ixl = random.randint(0, ncap - self.seq_per_img)
            seq_batch = cap_seq[ixl:ixl+self.seq_per_img,:,:4]
            mask_batch = box_mask[ixl:ixl+self.seq_per_img]

        input_seq = np.zeros([self.seq_per_img, self.seq_length+1, 4])
        input_seq[:,1:] = seq_batch

        gt_seq = np.zeros([10, self.seq_length])
        gt_seq[:ncap,:] = cap_seq[:,:,4]

        # load the image for visualization purposes
        if self.vis_attn:
            seg_show = np.zeros((self.num_sampled_frm, 1280, 720, 3))
            seg_dim_info = torch.LongTensor(2)
            for i in range(self.num_sampled_frm):
                try:
                    img = Image.open(os.path.join(self.opt.image_path, seg_id, str(i+1).zfill(2)+'.jpg')).convert('RGB')
                    width, height = img.size
                    seg_show[i, :height, :width] = np.array(img)
                    seg_dim_info[0] = height
                    seg_dim_info[1] = width
                except:
                    print('cannot load image...')
                    break
            seg_show = torch.from_numpy(seg_show).type(torch.ByteTensor)

        # padding the proposals and gt_bboxs
        pad_proposals = np.zeros((self.max_proposal, 7))
        pad_pnt_mask = np.ones((self.max_proposal))
        pad_gt_bboxs = np.zeros((self.max_gt_box, 6))
        pad_box_mask = np.ones((self.seq_per_img, self.max_gt_box, self.seq_length+1))
        pad_region_feature = np.zeros((self.max_proposal, self.att_feat_size))
        pad_frm_mask = np.ones((self.max_proposal, self.max_gt_box)) # mask out proposals outside the target frames

        num_box = min(gt_bboxs.shape[0], self.max_gt_box)
        num_pps = min(proposals.shape[0], self.max_proposal)
        pad_proposals[:num_pps] = proposals[:num_pps]
        pad_pnt_mask[:num_pps] = pnt_mask[:num_pps]
        pad_gt_bboxs[:num_box] = gt_bboxs[:num_box]
        pad_box_mask[:,:num_box,1:] = mask_batch[:,:num_box,:]
        pad_region_feature[:num_pps] = region_feature[:num_pps]

        frm_mask = self.get_frm_mask(pad_proposals[:num_pps, 4], pad_gt_bboxs[:num_box, 4])
        pad_frm_mask[:num_pps, :num_box] = frm_mask

        input_seq = torch.from_numpy(input_seq).long()
        gt_seq = torch.from_numpy(gt_seq).long()
        pad_proposals = torch.from_numpy(pad_proposals).float()
        pad_pnt_mask = torch.from_numpy(pad_pnt_mask).byte()
        pad_gt_bboxs = torch.from_numpy(pad_gt_bboxs).float()
        pad_box_mask = torch.from_numpy(pad_box_mask).byte()
        pad_region_feature = torch.from_numpy(pad_region_feature).float()
        pad_proposals.masked_fill_(pad_pnt_mask.view(-1, 1), 0.)
        pad_region_feature.masked_fill_(pad_pnt_mask.view(-1, 1), 0.)
        pad_frm_mask = torch.from_numpy(pad_frm_mask).byte()
        num = torch.FloatTensor([ncap, num_pps, num_box, int(seg_id_ix),
            max(self.num_seg_per_vid[vid_id_ix])+1, timestamps[0]*1./dur,
            timestamps[1]*1./dur]) # 3 + 4 (seg_id, num_of_seg_in_video, seg_start_time, seg_end_time)
        sample_idx = torch.from_numpy(sample_idx).long()

        if self.vis_attn:
            return seg_feature, input_seq, gt_seq, num, pad_proposals, pad_gt_bboxs, pad_box_mask, seg_id, seg_show, seg_dim_info, pad_region_feature, pad_frm_mask, sample_idx, pad_pnt_mask
        else:
            return seg_feature, input_seq, gt_seq, num, pad_proposals, pad_gt_bboxs, pad_box_mask, seg_id, pad_region_feature, pad_frm_mask, sample_idx, pad_pnt_mask


    def __len__(self):
        return len(self.split_ix)
