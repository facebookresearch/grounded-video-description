# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Last modified by Luowei Zhou on 09/25/2018

import os
import json
import argparse
from random import shuffle, seed
import string
import h5py
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
import pdb
from stanfordcorenlp import StanfordCoreNLP
from nltk.tokenize import word_tokenize

nlp = StanfordCoreNLP('tools/stanford-corenlp-full-2018-02-27')
props={'annotators': 'ssplit, tokenize, lemma','pipelineLanguage':'en', 'outputFormat':'json'}

def build_vocab(vids, split, params):
  count_thr = params['word_count_threshold']

  # count up the number of words
  # stats on sentence length distribution
  counts = {}
  sent_lengths = {}
  for vid_id, vid in vids.items():
      if split[vid_id] in ('training', 'validation'):
          for seg_id, seg in vid['segments'].items():
              nw = len(seg['tokens'])
              sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
              for w in seg['tokens']:
                  counts[w] = counts.get(w, 0)+1

  cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
  print('top words and their counts:')
  print('\n'.join(map(str,cw[:20])))

  print('The counts of empty token is {}'.format(counts['']))
  counts[''] = 0
  # print some stats
  total_words = sum(counts.values())
  print('total words:', total_words)
  bad_words = [w for w,n in counts.items() if n <= count_thr]
  vocab = [w for w,n in counts.items() if n > count_thr]
  bad_count = sum(counts[w] for w in bad_words)
  print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
  print('number of words in vocab would be %d' % (len(vocab), ))
  print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

  max_len = max(sent_lengths.keys())
  print('max length sentence in raw data: ', max_len)
  print('sentence length distribution (count, number of words):')
  sum_len = sum(sent_lengths.values())
  for i in range(max_len+1):
    print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

  # lets now produce the final annotations
  if bad_count > 0:
    # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    vocab.append('UNK')
  
  vids_new = {}
  for vid_id, vid in vids.items():
      # if split[vid_id] in ('training', 'validation'):
      if vid_id in split:
          segs_new = {}
          for seg_id, seg in vid['segments'].items():
              txt = seg['tokens']
              clss = seg['process_clss']
              bbox = seg['process_bnd_box']
              idx = seg['process_idx']
              frm_idx = seg['frame_ind']
              caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
              segs_new[seg_id] = {'caption':caption, 'clss':clss, 'bbox':bbox, 'idx':idx, 'frm_idx':frm_idx}
      vids_new[vid_id] = {}
      vids_new[vid_id]['segments'] = segs_new
      # vids_new[vid_id]['rwidth'] = vid['rwidth']
      # vids_new[vid_id]['rheight'] = vid['rheight']

  return vocab, vids_new

def main(params):

  imgs_split = json.load(open(params['split_file'], 'r'))
  split = {}
  for s, ids in imgs_split.items():
      for i in ids:
          split[i] = s # video names are the ids

  vids_processed = json.load(open(params['input_json'], 'r'))

  # word to detection label
  anet_class_all = vids_processed['vocab']
  wtod = {}
  for i in range(len(anet_class_all)):
      # TODO, assume each object class is a 1-gram, can try n-gram or multiple phrases later
      wtod[anet_class_all[i]] = i

  vids_processed = vids_processed['annotations']
  for vid_id, vid in vids_processed.items():
    if vid_id in split:
      vid['split'] = split[vid_id]
    else:
      vid['split'] = 'rest'
      print('Video {} can not be found in the dataset!'.format(vid_id))
  seed(123) # make reproducible

  # create the vocab
  vocab, vids_new = build_vocab(vids_processed, split, params)
  itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
  wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

  wtol = {}
  for w in vocab:
      out = json.loads(nlp.annotate(w.encode('utf-8'), properties=props))
      lemma_w = out['sentences'][0]['tokens'][0]['lemma']
      wtol[w] = lemma_w

  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['wtod'] = wtod
  out['wtol'] = wtol
  out['videos'] = []
  for vid_id, vid in vids_processed.items():
      jvid = {}
      jvid['vid_id'] = vid_id
      jvid['split'] = vid['split']
      seg_lst = vid['segments'].keys()
      seg_lst = [int(s) for s in seg_lst]
      seg_lst.sort()
      for i in seg_lst:
          jvid['seg_id'] = str(i)
          jvid['id'] = vid_id+'_segment_'+str(i).zfill(2) # some info
          out['videos'].append(jvid.copy())
  print('Total number of segments: {}'.format(len(out['videos'])))
  
  json.dump(out, open(params['output_dic_json'], 'w'))
  print('wrote ', params['output_dic_json'])

  json.dump(vids_new, open(params['output_cap_json'], 'w'))
  print('wrote ', params['output_cap_json'])

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--split_file', default='data/anet/split_ids_anet_entities.json')
  parser.add_argument('--input_json', default='data/anet/anet_cleaned_class_thresh50.json')
  parser.add_argument('--output_dic_json', default='data/anet/dic_anet.json', help='output json file')
  parser.add_argument('--output_cap_json', default='data/anet/cap_anet.json', help='output json file')

  # options
  parser.add_argument('--max_length', default=20, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=3, type=int, help='only words that occur more than this number of times will be put in vocab')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)
  nlp.close()
