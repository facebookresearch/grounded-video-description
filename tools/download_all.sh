#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Script to download all the necessary data files and place under the data directory
# Written by Luowei Zhou on 05/01/2019


DATA_ROOT='data'

mkdir -p $DATA_ROOT/flickr30k save results log

# annotation files
wget -P $DATA_ROOT/flickr30k http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
wget -P $DATA_ROOT/flickr30k https://www.dropbox.com/s/twve5exs8qj9xgd/flickr30k.tar.gz
wget -P tools/coco-caption/annotations https://github.com/jiasenlu/coco-caption/raw/master/annotations/caption_flickr30k.json
wget -P $DATA_ROOT/flickr30k https://www.dropbox.com/s/h4ru86ocb10axa1/flickr30k_cleaned_class.json.tar.gz

# feature files
wget -P $DATA_ROOT https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz
wget -P $DATA_ROOT/imagenet_weights https://www.dropbox.com/sh/67fc8n6ddo3qp47/AAACkO4QntI0RPvYic5voWHFa/resnet101.pth
wget -P $DATA_ROOT/flickr30k https://dl.fbaipublicfiles.com/ActivityNet-Entities/flickr30k-Entities/flickr30k_detection_vg_X-101-64x4d-FPN_2x_feature.tar.gz
wget -P $DATA_ROOT/flickr30k https://dl.fbaipublicfiles.com/ActivityNet-Entities/flickr30k-Entities/flickr30k_detection_vg_X-101-64x4d-FPN_2x_feat_map_100prop_box_only.h5

# Stanford CoreNLP 3.9.1
wget -P tools/ http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip

# pre-trained models
wget -P save/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/flickr30k-Entities/flickr30k-Entities_pre-trained-models.tar.gz

# uncompress
cd $DATA_ROOT
for file in *.tar.gz; do tar -zxvf "${file}" && rm "${file}"; done
cd flickr30k
for file in *.tar.gz; do tar -zxvf "${file}" && rm "${file}"; done
unzip -j caption_datasets.zip dataset_flickr30k.json -d . && rm caption_datasets.zip
tar -zxvf flickr30k.tar.gz && mv flickr30k/* . && rm -r flickr30k
cd ../../tools
for file in *.zip; do unzip "${file}" && rm "${file}"; done
cd coco-caption
./get_stanford_models.sh
cd ../../save
for file in *.tar.gz; do tar -zxvf "${file}" && rm "${file}"; done
mv flickr30k-Entities_pre-trained-models/* . && rm -r flickr30k-Entities_pre-trained-models
