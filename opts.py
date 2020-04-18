# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--path_opt', type=str, default='cfgs/anet_res101_vg_feat_10x100prop.yml',
                    help='')
    parser.add_argument('--dataset', type=str, default='anet',
                    help='')
    parser.add_argument('--input_json', type=str, default='',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_dic', type=str, default='',
                    help='path to the json containing the preprocessed dataset')
    parser.add_argument('--image_path', type=str, default='',
                    help='path to the h5file containing the image data') 
    parser.add_argument('--proposal_h5', type=str, default='',
                    help='path to the json containing the detection result.') 
    parser.add_argument('--feature_root', type=str, default='',
                    help='path to the npy flies containing region features')
    parser.add_argument('--seg_feature_root', type=str, default='',
                    help='path to the npy files containing frame-wise features')

    parser.add_argument('--num_workers', type=int, default=20,
                    help='number of worker to load data')
    parser.add_argument('--cuda', action='store_true',
                    help='whether use cuda')
    parser.add_argument('--mGPUs', action='store_true',
                    help='whether use multiple GPUs')

    # Model settings
    parser.add_argument('--rnn_size', type=int, default=1024,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=3072,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--t_attn_size', type=int, default=480, help='number of frames sampled for temopral attention')
    parser.add_argument('--num_sampled_frm', type=int, default=10)
    parser.add_argument('--num_prop_per_frm', type=int, default=100)
    parser.add_argument('--prop_thresh', type=float, default=0.2,
                    help='threshold to filter out low-confidence proposals')

    parser.add_argument('--att_model', type=str, default='topdown',
                    help='different attention model, now supporting topdown | transformer(unsupervised)')
    parser.add_argument('--att_input_mode', type=str, default='both',
                    help='use whether featmap|region|dual_region|both in topdown language model')
    parser.add_argument('--t_attn_mode', type=str, default='bigru',
                    help='temporal attention context encoding mode: bilstm | bigru')
    parser.add_argument('--transfer_mode', type=str, default='cls', help='knowledge transfer mode, could be cls|glove|both')
    parser.add_argument('--region_attn_mode', type=str, default='mix',
                    help='options: dp|add|cat|mix, dp stands for dot-product, add for additive, cat for concat, mix indicates dp for grd. and add for attn., mix_mul indicates dp for grd. and element-wise multiplication for attn.')

    parser.add_argument('--enable_BUTD', action='store_true', help='if enable, the region feature will not include location embedding nor class encoding')
    parser.add_argument('--obj_interact', action='store_true', help='self-attention encoding for region features')
    parser.add_argument('--exclude_bgd_det', action='store_true', help='exclude __background__ RoIs')

    parser.add_argument('--w_att2', type=float, default=0)
    parser.add_argument('--w_grd', type=float, default=0)
    parser.add_argument('--w_cls', type=float, default=0)
    parser.add_argument('--disable_caption', action='store_true', help='set to disable caption generation loss')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=40,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=10,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--seq_per_img', type=int, default=1,
                    help='number of captions to sample for each image during training')
    parser.add_argument('--seq_length', type=int, default=20, help='')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')

    # set training session
    parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')

    # Evaluation/Checkpointing
    parser.add_argument('--train_split', type=str, default='training',
                    help='')
    parser.add_argument('--val_split', type=str, default='validation',
                    help='')
    parser.add_argument('--inference_only', action='store_true',
                    help='')
    parser.add_argument('--densecap_references', type=str, nargs='+', default=['./data/anet/anet_entities_val_1.json', './data/anet/anet_entities_val_2.json'],
                        help='reference files with ground truth captions to compare results against. delimited (,) str')
    parser.add_argument('--densecap_verbose', action='store_true', help='evaluate CIDEr only or all language metrics in densecap')
    parser.add_argument('--grd_reference', type=str, default='tools/anet_entities/data/anet_entities_cleaned_class_thresh50_trainval.json')
    parser.add_argument('--split_file', type=str, default='tools/anet_entities/data/split_ids_anet_entities.json')

    parser.add_argument('--eval_obj_grounding_gt', action='store_true',
                    help='whether evaluate object grounding accuracy')
    parser.add_argument('--eval_obj_grounding', action='store_true',
                    help='whether evaluate object grounding accuracy')
    parser.add_argument('--vis_attn', action='store_true', help='visualize attention')
    parser.add_argument('--enable_visdom', action='store_true')
    parser.add_argument('--visdom_server', type=str, default='', help='update it with your server url')

    parser.add_argument('--val_images_use', type=int, default=5000,
                    help='how many segments to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--val_every_epoch', type=int, default=2,
                    help='how many segments to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', action='store_true',
                    help='Evaluate language as well (1 = yes, 0 = no)?')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       
    parser.add_argument('--disp_interval', type=int, default=100,
                    help='how many iteration to display an loss.')       
    parser.add_argument('--losses_log_every', type=int, default=10,
                    help='how many iteration for log.')
    parser.add_argument('--det_oracle', action='store_true',
                    help='whether use oracle bounding box.')
    parser.add_argument('--frm_oracle', action='store_true',
                    help='whether use oracle frame.')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    return args
