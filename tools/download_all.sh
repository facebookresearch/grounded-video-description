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

mkdir -p $DATA_ROOT/anet save results log

# annotation files
wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_prep.tar.gz
wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_captions.tar.gz
wget -P tools/coco-caption/annotations https://github.com/jiasenlu/coco-caption/raw/master/annotations/caption_flickr30k.json

# feature files
wget -P $DATA_ROOT/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz
wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/rgb_motion_1d.tar.gz
wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_detection_vg_fc6_feat_100rois.h5
wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/fc6_feat_100rois.tar.gz

# Stanford CoreNLP 3.9.1
wget -P tools/ http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip

# pre-trained models
wget -P save/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/pre-trained-models.tar.gz

# uncompress
cd $DATA_ROOT
for file in *.tar.gz; do tar -zxvf "${file}" && rm "${file}"; done
cd anet
for file in *.tar.gz; do tar -zxvf "${file}" && rm "${file}"; done
cd ../../tools
for file in *.zip; do unzip "${file}" && rm "${file}"; done
cd coco-caption
./get_stanford_models.sh
cd ../../save
for file in *.tar.gz; do tar -zxvf "${file}" && rm "${file}"; done
mv pre-trained-models/* . && rm -r pre-trained-models
