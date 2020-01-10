# Grounded Video Description (flickr30k)

### If you like this work, or image grounding and captioning in general, you might want to check out our recent work on vision-language pre-training ([VLP](https://github.com/LuoweiZhou/VLP))!

This repo hosts the source code for our paper [Grounded Video Description](https://arxiv.org/pdf/1812.06587.pdf). It supports [Flickr30k-Entities](https://github.com/BryanPlummer/flickr30k_entities) dataset. We also have code that supports [ActivityNet-Entities](https://github.com/facebookresearch/ActivityNet-Entities) dataset, hosted at the [Master](https://github.com/facebookresearch/grounded-video-description) branch.


## Quick Start
### Preparations
Follow the instructions 1 to 3 in the [Requirements](#req) section to install required packages.

### Download everything
We provide a script to download most of the data you need, including all the annotations, feature files and pre-trained models (total 14GB):
```
bash tools/download_all.sh
```
The only exception is the Flickr30k images which you will need to fill out a form at the bottom of this [page](http://hockenmaier.cs.illinois.edu/DenotationGraph/) to get. Then, link the image root to `data/flickr30k/images`.

### Starter code
Run the following eval code to test if your environment is setup:
```
python main.py --batch_size 50 --cuda --num_workers 10 --max_epoch 50 --inference_only \
    --start_from save/flickr-sup-0.1-0.1-0.1-run1 --id flickr-sup-0.1-0.1-0.1-run1 \
    --seq_length 20 --language_eval --eval_obj_grounding --obj_interact
```

(Optional) Single-GPU training code for double-check:
```
python main.py --batch_size 20 --cuda --checkpoint_path save/gvd_starter --id gvd_starter --language_eval
```
You can now skip to the [Training and Validation](#train) section!


## <a name='req'></a> Requirements (Recommended)
1) Clone the repo recursively:
```
git clone -b flickr_branch --recursive git@github.com:facebookresearch/grounded-video-description.git
```
Make sure the submodule [coco-caption](https://github.com/tylin/coco-caption) is included.

2) Install CUDA 9.0 and CUDNN v7.1. Later versions should be fine, but might need to get the conda env file updated (e.g., for PyTorch).

3) Install [Miniconda](https://conda.io/miniconda.html) (either Miniconda2 or 3, version 4.6+). We recommend using conda [environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to install required packages, including Python 2.7, [PyTorch 1.1.0](https://pytorch.org/get-started/locally/) etc.:
```
MINICONDA_ROOT=[to your Miniconda root directory]
conda env create -f cfgs/conda_env_gvd.yml --prefix $MINICONDA_ROOT/envs/gvd_pytorch1.1
conda activate gvd_pytorch1.1
```

4) (Optional) If you choose to not use `download_all.sh`, be sure to install JAVA and download Stanford CoreNLP for SPICE (see [here](https://github.com/tylin/coco-caption)). Also, download and place the reference [file](https://github.com/jiasenlu/coco-caption/blob/master/annotations/caption_flickr30k.json) under `coco-caption/annotations`. Download [Stanford CoreNLP 3.9.1](https://stanfordnlp.github.io/CoreNLP/history.html) for grounding evaluation and place the uncompressed folder under the `tools` directory.


## Data Preparation
Download the dataset images from [here](https://github.com/BryanPlummer/flickr30k_entities) and preprocessed Karpathy's split of Flickr30k from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). Extract and place `dataset_flickr30k.json` from the zip file under `data/flickr30k`. Download preprocessed Flickr30k annotation from [NeuralBabyTalk](https://www.dropbox.com/s/twve5exs8qj9xgd/flickr30k.tar.gz?dl=1), uncompress and place under the same folder.

The region features and detections are available for download ([feature](https://dl.fbaipublicfiles.com/ActivityNet-Entities/flickr30k-Entities/flickr30k_detection_vg_X-101-64x4d-FPN_2x_feature.tar.gz) and [detection](https://dl.fbaipublicfiles.com/ActivityNet-Entities/flickr30k-Entities/flickr30k_detection_vg_X-101-64x4d-FPN_2x_feat_map_100prop_box_only.h5)). The region feature file should be decompressed and placed under your feature directory. We refer to the region feature directory as `feature_root` in the code. The H5 region detection (proposal) file is referred to as `proposal_h5` in the code.

Other auxiliary files, such as the weights from Detectron fc7 layer, are available [here](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz). Uncompress and place under the `data` directory.


## <a name='train'></a> Training and Validation
Modify the config file `cfgs/flickr30k_res101_vg_feat_100prop.yml` with the correct dataset and feature paths (or through simlinks). Create new directories `log` and `results` under the root directory to save log and result files.

The example command on running a 8-GPU data parallel job:

For supervised models (with self-attention):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --path_opt cfgs/flickr30k_res101_vg_feat_100prop.yml \
    --batch_size $batch_size --cuda --checkpoint_path save/$ID --id $ID --mGPUs \
    --language_eval --w_att2 $w_att2 --w_grd $w_grd --w_cls $w_cls --obj_interact | tee log/$ID
```

For unsupervised models (without self-attention):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --path_opt cfgs/flickr30k_res101_vg_feat_100prop.yml \
    --batch_size $batch_size --cuda --checkpoint_path save/$ID --id $ID --mGPUs \
    --language_eval | tee log/$ID
```
Arguments: `batch_size=240`, `w_att2=0.1`, `w_grd=0.1`, `w_cls=0.1`, `ID` indicates the model name. (Optional) Remove `--mGPUs` to run in single-GPU mode. 

### Pre-trained Models
The pre-trained models can be downloaded from [here (2.5GB)](https://dl.fbaipublicfiles.com/ActivityNet-Entities/flickr30k-Entities/flickr30k-Entities_pre-trained-models.tar.gz). Make sure you uncompress the file under the `save` directory (create one under the root directory if not exists).


## Inference and Testing
For supervised models (`ID=flickr-sup-0.1-0.1-0.1-run1`):

(standard inference: language evaluation and localization evaluation on generated sentences)

```
python main.py --path_opt cfgs/flickr30k_res101_vg_feat_100prop.yml --batch_size 50 --cuda \
    --num_workers 10 --max_epoch 50 --inference_only --start_from save/$ID --id $ID \
    --val_split $val_split --seq_length 20 --language_eval --eval_obj_grounding --obj_interact \
    | tee log/eval-$split-$ID-beam$beam_size-standard-inference
```

(GT inference: localization evaluation on GT sentences)

```
python main.py --path_opt cfgs/flickr30k_res101_vg_feat_100prop.yml --batch_size 50 --cuda \
    --num_workers 10 --max_epoch 50 --inference_only --start_from save/$ID --id $ID \
    --val_split $val_split --seq_length 40 --eval_obj_grounding_gt --obj_interact \
    | tee log/eval-$split-$ID-beam$beam_size-gt-inference
```

For unsupervised models (`ID=flickr-unsup-0-0-0-run1`), simply remove the `--obj_interact` option.

Arguments: `val_split='val' or 'test'`.


## Visualization
We refer to the Flickr30k image folder as `image_path`. During validation or testing, add the `--vis_attn` option and the visualization for randomly sampled images will be placed under the `vis` directory.


## Reference
Please acknowledge the following paper if you use the code:

```
@inproceedings{zhou2019grounded,
  title={Grounded Video Description},
  author={Zhou, Luowei and Kalantidis, Yannis and Chen, Xinlei and Corso, Jason J and Rohrbach, Marcus},
  booktitle={CVPR},
  year={2019}
}
```


## Acknowledgement
We thank Jiasen Lu for his [Neural Baby Talk](https://github.com/jiasenlu/NeuralBabyTalk) repo. We thank Chih-Yao Ma for his helpful discussions.


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree. 

Portions of the source code are based on the [Neural Baby Talk](https://github.com/jiasenlu/NeuralBabyTalk) project.
