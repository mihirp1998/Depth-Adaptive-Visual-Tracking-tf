
"""Default configurations of model specification, training and tracking

For most of the time, DO NOT modify the configurations within this file.
Use the configurations here as the default configurations and only update
them following the examples in the `experiments` directory.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp

LOG_DIR = 'Logs/SiamFC'  # where checkpoints, logs are saved
RUN_NAME = 'SiamFC-3s-color-scratch'  # identifier of the experiment

MODEL_CONFIG = {
  'z_image_size': 127,  # Exemplar image size

  'embed_config': {'embedding_name': 'convolutional_alexnet',
                   'embedding_checkpoint_file': None,  # mat file path of the pretrained embedding model.
                   'train_embedding': True,
                   'init_method': 'kaiming_normal',
                   'use_bn': True,
                   'bn_scale': True,
                   'bn_momentum': 0.05,
                   'bn_epsilon': 1e-6,
                   'embedding_feature_num': 256,
                   'weight_decay': 5e-4,
                   'stride': 8, },

  'adjust_response_config': {'train_bias': True,
                             'scale': 1e-3, },
}

TRAIN_CONFIG = {
  'train_dir': osp.join(LOG_DIR, 'track_model_checkpoints', RUN_NAME),

  'seed': 123,  # fix seed for reproducing experiments

  'train_data_config': {'input_imdb': 'data/train_imdb.pickle',
                        'preprocessing_name': 'siamese_fc_color',
                        'num_examples_per_epoch': 5.32e4,
                        'epoch': 50,
                        'batch_size': 1,
                        'max_frame_dist': 100,  # Maximum distance between any two random frames draw from videos.
                        'prefetch_threads': 4,
                        'prefetch_capacity': 15 * 8, },  # The maximum elements number in the data loading queue

  'validation_data_config': {'input_imdb': 'data/validation_imdb.pickle',
                             'preprocessing_name': 'None',
                             'batch_size': 8,
                             'max_frame_dist': 100,  # Maximum distance between any two random frames draw from videos.
                             'prefetch_threads': 1,
                             'prefetch_capacity': 15 * 8, },  # The maximum elements number in the data loading queue

  # Configurations for generating groundtruth maps
  'gt_config': {'rPos': 16,
                'rNeg': 0, },

  # Optimizer for training the model.
  'optimizer_config': {'optimizer': 'MOMENTUM',  # SGD and MOMENTUM are supported
                       'momentum': 0.9,
                       'use_nesterov': False, },

  # Learning rate configs
  'lr_config': {'policy': 'exponential',
                'initial_lr': 0.01,
                'num_epochs_per_decay': 1,
                'lr_decay_factor': 0.8685113737513527,
                'staircase': True, },

  # If not None, clip gradients to this value.
  'clip_gradients': None,

  # Frequency at which loss and global step are logged
  'log_every_n_steps': 10,

  # Frequency to save model
  'save_model_every_n_step': 5.32e4 // 8,  # save model every epoch

  # How many model checkpoints to keep. No limit if None.
  'max_checkpoints_to_keep': None,
}
