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
from PIL import Image
from torchvision.datasets.folder import default_loader
import torch
import torch.utils.data as data
import copy
import misc.utils as utils
from PIL import Image
import torchvision.transforms as transforms
import torchtext.vocab as vocab # use this to load glove vector

class DataLoader(data.Dataset):
    def __init__(self, opt, split='train', seq_per_img=5):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.seq_length = opt.seq_length
        self.split = split
        self.aug_gt_det = opt.aug_gt_det
        self.seq_per_img = seq_per_img
        self.att_feat_size = opt.att_feat_size
        self.vis_attn = opt.vis_attn
        self.feature_root = opt.feature_root
        # image processing function.
        if split == 'train':
            self.Resize = transforms.Resize((self.opt.image_size, self.opt.image_size))
        else:
            self.Resize = transforms.Resize((self.opt.image_crop_size, self.opt.image_crop_size))
        self.RandomCropWithBbox = utils.RandomCropWithBbox(opt.image_crop_size)
        self.ToTensor = transforms.ToTensor()
        self.res_Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.vgg_pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]])
        self.exclude_bgd_det = opt.exclude_bgd_det
        self.prop_thresh = opt.prop_thresh

        self.max_gt_box = 100
        self.max_proposal = 200
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
        print('DataLoader loading json file: ', opt.input_json)
        self.caption_file = json.load(open(self.opt.input_json))

        # open the detection json file.
        print('DataLoader loading proposal file: ', opt.proposal_h5)
        h5_proposal_file = h5py.File(self.opt.proposal_h5, 'r', driver='core')
        self.label_proposals = h5_proposal_file['dets_labels'][:]
        self.num_nms = h5_proposal_file['nms_num'][:]
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

        if self.aug_gt_det:
            self.glove_vg_cls = np.concatenate((self.glove_clss, self.glove_vg_cls), axis=0)

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
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == split:
                if opt.vis_attn:
                    if random.random() < 0.01: # randomly sample 1% segments to visualize
                        self.split_ix.append(ix)
                else:
                    self.split_ix.append(ix)
        print('assigned %d images to split %s' %(len(self.split_ix), split))

    def get_det_word(self, gt_bboxs, captions):
        
        # get the present category.
        pcats = []
        for i in range(gt_bboxs.shape[0]):
            pcats.append(gt_bboxs[i,5])
        # get the orginial form of the caption.
        indicator = []
        for i, sent in enumerate(captions):
            indicator.append([(0, 0, 0)]*len(sent['caption'])) # category class, binary class, fine-grain class.
            for j, bbox_idx in enumerate(sent['bbox_idx']):
                # if the bbox_idx is not filtered out.
                if bbox_idx in pcats:
                    w_idx = sent['idx'][j]
                    ng = sent['clss'][j]
                    bn = (ng != sent['caption'][w_idx]) + 1
                    fg = self.dtoi[ng]
                    indicator[i][w_idx] = (self.wtod[sent['clss'][j]], bn, fg)
        
        return indicator

    def __getitem__(self, index):

        ix = self.split_ix[index]

        # load image here.
        image_id = self.info['images'][ix]['id']
        file_path = self.info['images'][ix]['file_path']

        # load the proposal file
        num_nms = int(self.num_nms[ix])
        proposals = copy.deepcopy(self.label_proposals[ix])
        proposals = proposals[:num_nms,:]

        region_feature = np.load(os.path.join(self.feature_root, str(image_id)+'.npy'))

        # filter out low-confidence proposals or backgrounds
        prop_thresh_mask = (proposals[:, 5] > self.prop_thresh)
        if self.exclude_bgd_det:
            non_background_idx = np.nonzero(proposals[:, 4]*prop_thresh_mask)[0]
        else:
            non_background_idx = np.nonzero(prop_thresh_mask)[0]
        proposals = proposals[non_background_idx, :] # exclude __background__ detections
        region_feature = region_feature[non_background_idx, :]
        num_nms = non_background_idx.shape[0]

        region_feature = region_feature[:num_nms, :]
        captions = copy.deepcopy(self.caption_file[ix])

        bbox_ann = []
        bbox_idx = 0
        for cap_idx, sent in enumerate(captions):
            sent['bbox_idx'] = []
            for i, box in enumerate(sent['bbox']):
                if sent['idx'][i] < self.seq_length: # we don't care about the boxes outside the length limit.
                                                     # after all our goal is referring, not detection
                    sent['bbox_idx'].append(bbox_idx)
                    bbox_ann.append({'bbox':box, 'label': self.dtoi[sent['clss'][i]], 'bbox_idx':bbox_idx, \
                        'idx':sent['idx'][i], 'cap_idx':cap_idx})
                    bbox_idx += 1

        gt_bboxs = np.zeros((len(bbox_ann), 8))
        for i, bbox in enumerate(bbox_ann):
            gt_bboxs[i, :4] = bbox['bbox']
            gt_bboxs[i, 4] = bbox['label']
            gt_bboxs[i, 5] = bbox['bbox_idx']
            gt_bboxs[i, 6] = bbox['idx']
            gt_bboxs[i, 7] = bbox['cap_idx']

        # load the image.
        img = Image.open(os.path.join(self.opt.image_path, file_path)).convert('RGB')
        width, height = img.size

        # resize the image.
        img = self.Resize(img)

        # resize the gt_bboxs and proposals.
        # ATTENTION, unlike NBT, we do not dynamically crop regions from feature map during training
        if self.split == 'train':
            proposals = utils.resize_bbox(proposals, width, height, self.opt.image_size, self.opt.image_size)
            gt_bboxs = utils.resize_bbox(gt_bboxs, width, height, self.opt.image_size, self.opt.image_size)
        else:
            proposals = utils.resize_bbox(proposals, width, height, self.opt.image_crop_size, self.opt.image_crop_size)
            gt_bboxs = utils.resize_bbox(gt_bboxs, width, height, self.opt.image_crop_size, self.opt.image_crop_size)

        img, proposals, gt_bboxs = self.RandomCropWithBbox(img, proposals, gt_bboxs) # random crop img

        gt_x = (gt_bboxs[:,2]-gt_bboxs[:,0]+1)
        gt_y = (gt_bboxs[:,3]-gt_bboxs[:,1]+1)
        gt_area_nonzero = (((gt_x != 1) & (gt_y != 1)))

        gt_bboxs = gt_bboxs[gt_area_nonzero]

        # given the bbox_ann, and caption, this function determine which word belongs to the detection.
        det_indicator = self.get_det_word(gt_bboxs, captions)
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
            box_mask[int(gt_bboxs[i][7]), i, int(gt_bboxs[i][6])] = 0

        gt_bboxs = gt_bboxs[:,:5]

        # get the batch version of the seq and box_mask.
        if ncap < self.seq_per_img:
            seq_batch = np.zeros([self.seq_per_img, self.seq_length, 4])
            mask_batch = np.zeros([self.seq_per_img, gt_bboxs.shape[0], self.seq_length])
            # we need to subsample (with replacement)
            for q in range(self.seq_per_img):
                ixl = random.randint(0,ncap)
                seq_batch[q,:] = cap_seq[ixl,:,:4]
                mask_batch[q,:]=box_mask[ixl]
        else:
            ixl = random.randint(0, ncap - self.seq_per_img)
            seq_batch = cap_seq[ixl:ixl+self.seq_per_img,:,:4]
            mask_batch = box_mask[ixl:ixl+self.seq_per_img]

        input_seq = np.zeros([self.seq_per_img, self.seq_length+1, 4])
        input_seq[:,1:] = seq_batch

        gt_seq = np.zeros([10, self.seq_length])
        gt_seq[:ncap,:] = cap_seq[:,:,4]

        img_show = np.array(img)

        # padding the proposals and gt_bboxs
        pad_proposals = np.zeros((self.max_proposal, 6))
        pad_gt_bboxs = np.zeros((self.max_gt_box, 5))
        pad_box_mask = np.ones((self.seq_per_img, self.max_gt_box, self.seq_length+1))
        pad_region_feature = np.zeros((self.max_proposal, self.att_feat_size))

        if self.aug_gt_det:
            proposals[:, 4] += self.glove_clss.shape[0]
            if self.split == 'train':
                # augment the proposal with the gt bounding box. confident score is 1
                gt_bboxs_tmp = np.concatenate((gt_bboxs, np.ones((gt_bboxs.shape[0],1))), axis=1)
                proposals = np.concatenate((gt_bboxs_tmp, proposals), axis=0)

        num_pps = min(proposals.shape[0], self.max_proposal)
        num_box = min(gt_bboxs.shape[0], self.max_gt_box)
        pad_proposals[:num_pps] = proposals[:num_pps]
        pad_gt_bboxs[:num_box] = gt_bboxs[:num_box]
        pad_box_mask[:,:num_box,1:] = mask_batch[:,:num_box,:]
        pad_region_feature[:num_pps] = region_feature[:num_pps]

        input_seq = torch.from_numpy(input_seq).long()
        gt_seq = torch.from_numpy(gt_seq).long()
        pad_proposals = torch.from_numpy(pad_proposals).float()
        pad_box_mask = torch.from_numpy(pad_box_mask).byte()
        pad_gt_bboxs = torch.from_numpy(pad_gt_bboxs).float()
        pad_region_feature = torch.from_numpy(pad_region_feature).float()
        num = torch.FloatTensor([ncap, num_pps, num_box, width, height])

        if self.opt.cnn_backend == 'vgg16':
            img = np.array(img, dtype='float32')
            img = img[:,:,::-1].copy() # RGB --> BGR
            img -= self.vgg_pixel_mean
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1).contiguous()
        else:
            img = self.ToTensor(img)
            img = self.res_Normalize(img)

        if self.vis_attn:
            return img, input_seq, gt_seq, num, pad_proposals, pad_gt_bboxs, pad_box_mask, image_id, img_show, pad_region_feature
        else:
            return img, input_seq, gt_seq, num, pad_proposals, pad_gt_bboxs, pad_box_mask, image_id, pad_region_feature

    def __len__(self):
        return len(self.split_ix)
