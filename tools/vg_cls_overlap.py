# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from stanfordcorenlp import StanfordCoreNLP
import torchtext

nlp = StanfordCoreNLP('./stanford-corenlp-full-2018-02-27')
props={'annotators': 'ssplit, tokenize, lemma','pipelineLanguage':'en', 'outputFormat':'json'}

# dataset we support: VG, ANet-Entities, Flickr30k-Entities, Open Image v4, Something-Something v2
src_dataset = 'VG'
grd_dataset = 'ss2'

freq_thresh = 100

def get_dataset_cls(src_dataset):
    classes = []
    if src_dataset == 'VG':
        classes = []
        obj_cls_file = 'vg_object_vocab.txt'
        with open(obj_cls_file) as f:
            data = f.readlines()
            classes.extend([i.strip() for i in data])
    elif src_dataset == 'flickr30k':
        obj_cls_file = 'data/flickr30k/flickr30k_class_name.txt'
        with open(obj_cls_file) as f:
            data = f.readlines()
            classes.extend([i.strip() for i in data])
    elif src_dataset == 'Open':
        cls_file_map = 'open_image_vocab_map.txt'
        vocab_dict = {}
        with open(cls_file_map) as f:
            data = f.readlines()
            for i in data:
                spt = i.split(',')
                idx = spt[0]
                cls = ','.join(spt[1:])
                vocab_dict[idx.strip().replace('"', '')] = cls.strip().replace(' ', '-').replace('"', '')

        print(vocab_dict)
        obj_cls_file = 'open_image_vocab.txt'
        with open(obj_cls_file) as f:
            data = f.readlines()
            classes.extend([vocab_dict[i.strip()] for i in data if vocab_dict.get(i.strip(), 'dummy') != 'dummy']) # fill in dummy if does not exist
    elif src_dataset == 'ss2':
        src_file_lst = ['something-something-v2-train.json', 'something-something-v2-validation.json']
        data = []
        for src_file in src_file_lst:
            with open(src_file) as f:
                data.extend(json.load(f))
        
        obj_vocab = {}
        for i in data:
            for cls in i['placeholders']:
                if cls in obj_vocab:
                    obj_vocab[cls] += 1
                else:
                    obj_vocab[cls] = 1
        print('unique object class in ss2: ', len(obj_vocab))
        print('top 100 frequent object class:', sorted(obj_vocab.keys(), key=lambda k:obj_vocab[k], reverse=True)[:100])
        classes = obj_vocab
    elif src_dataset == 'anet':
        src_file_lst = ['train.json', 'val_1.json']
        class_dict = {}
        for src_file in src_file_lst:
            with open(src_file) as f:
                data = json.load(f)
                for k, i in data.items():
                    for s in i['sentences']:
                        out = json.loads(nlp.annotate(s.encode('utf-8'), properties=props))
                        if len(out['sentences']) > 0:
                            for token in out['sentences'][0]['tokens']:
                                if 'NN' in token['pos']:
                                    lemma_w = token['lemma']
                                    if lemma_w in class_dict:
                                        class_dict[lemma_w] += 1
                                    else:
                                        class_dict[lemma_w] = 1

        tmp = {}
        for k, freq in class_dict.items():
            if freq >= freq_thresh:
                tmp[k] = freq
        class_dict = tmp
        print('number of lemma word in the vocab: ', len(class_dict), src_dataset)
        print('unique object class in ss2: ', len(tmp))
        print('top 100 frequent object class:', sorted(tmp.keys(), key=lambda k:tmp[k], reverse=True)[:100])
        return class_dict

    class_dict = {}
    if src_dataset != 'ss2':
        for i, w in enumerate(classes):
            w_s = w.split(',')
            for v in w_s:
                # if len(v.split(' ')) == 1:
                out = json.loads(nlp.annotate(v.encode('utf-8'), properties=props))
                if len(out['sentences']) > 0:
                    for token in out['sentences'][0]['tokens']:
                        if 'NN' in token['pos']:
                            lemma_w = token['lemma']
                            class_dict[lemma_w] = i
        print('number of lemma word in the vocab: ', len(class_dict), src_dataset)
    else:
        for w, freq in classes.items():
            w_s = w.split(',')
            for v in w_s:
                # if len(v.split(' ')) == 1:
                out = json.loads(nlp.annotate(v.encode('utf-8'), properties=props))
                if len(out['sentences']) > 0:
                    for token in out['sentences'][0]['tokens']:
                        if 'NN' in token['pos']:
                            lemma_w = token['lemma']
                            if lemma_w in class_dict:
                                class_dict[lemma_w] += freq
                            else:
                                class_dict[lemma_w] = freq

        tmp = {}
        for k, freq in class_dict.items():
            if freq >= freq_thresh:
                tmp[k] = freq
        class_dict = tmp
        print('number of lemma word in the vocab: ', len(class_dict), src_dataset)

    return class_dict


def load_corpus(grd_dataset):
    # vocab frequency
    sentences = []
    if grd_dataset == 'flickr30k':
        imgs_processed = json.load(open('../data/flickr30k/flickr30k_cleaned_class.json', 'r'))
        imgs_processed = imgs_processed['annotations']

        sentences = []
        for img in imgs_processed:
            for i in img['captions']:
                sentences.append(' '.join(i['tokens']))
    elif grd_dataset == 'ss2':
        src_file_lst = ['something-something-v2-train.json', 'something-something-v2-validation.json']
        data = []
        for src_file in src_file_lst:
            with open(src_file) as f:
                data.extend(json.load(f))

        for i in data:
            sentences.append(i['label'])
    elif grd_dataset == 'anet':
        src_file_lst = ['train.json', 'val_1.json']
        for src_file in src_file_lst:
            with open(src_file) as f:
                data = json.load(f)
                for k, i in data.items():
                    for s in i['sentences']:
                        sentences.append(s)
    else:
        raise NotImplementedError

    return sentences


def main():

    class_dict = get_dataset_cls(src_dataset)
    g_class_dict = get_dataset_cls(grd_dataset)
    text_proc = torchtext.data.Field(sequential=True, tokenize='spacy',
                                     lower=True, batch_first=True)
    sentences = load_corpus(grd_dataset)

    print('Total number of sentences: {}'.format(len(sentences)))
    sentences_proc = list(map(text_proc.preprocess, sentences))
    text_proc.build_vocab(sentences_proc)
    # print(text_proc.vocab.freqs.most_common(20))

    # check the overlapped vocab
    missed_cls = []
    catched_cls = []

    for k, i in g_class_dict.items():
        if k not in class_dict:
            missed_cls.append((k, text_proc.vocab.freqs[k]))
        else:
            catched_cls.append((k, text_proc.vocab.freqs[k]))

    missed_cls = sorted(missed_cls, key=lambda x:x[1], reverse=True)
    catched_cls = sorted(catched_cls, key=lambda x:x[1], reverse=True)

    for i, tup in enumerate(missed_cls):
        if i < 20:
            print('{}: {}'.format(tup[0], tup[1]))
    # for i, tup in enumerate(catched_cls):
    #     if i < 20:
    #         print('{}: {}'.format(tup[0], tup[1]))

    print('Number of classes are missing: {}, percentage: {}'.format(len(missed_cls),
          len(missed_cls)*1./len(g_class_dict)))


if __name__ == "__main__":
    main()
