
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import os.path as osp
import random
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import configuration
import sys

from model import compAdaptiveSiam
from utils.misc_utils import auto_select_gpu, mkdir_p, save_cfgs



def configurations():
  print("configuration")
  model_config = configuration.MODEL_CONFIG
  train_config = configuration.TRAIN_CONFIG
  track_config = configuration.TRACK_CONFIG
  return model_config,train_config,track_config

# function to train vgg weights and soft gates extractor weights
def train(model_config, train_config):

  print("running main")


  g = tf.Graph()
  with g.as_default():
    
    # seeds to be fix ...if exploding gradients change seeds

    random.seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    tf.set_random_seed(train_config['seed'])

    # This is the first Siamese Computation Adaptive model 
    model = compAdaptiveSiam.CompSiamModel(model_config, train_config, mode='train')


    # creates graphs and tensors for all operations pretrained vgg weights.. download link provided in readme
    model.build(vgg_pretrain="pretrained/new_vgg19.npy")


    siam_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="siamese")

    # l2 regularization for exploding gradient
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in siam_vars if 'bias' not in v.name]) * 0.01

    # train vgg net and gated weights seperately as mentioned in paper

    # vgg loss
    vgg_loss = model.total_loss + lossL2

    # gated loss
    gated_loss = model.total_gate_loss
    
    # vgg optimizer seperately done for clipping
    optimize_vgg_loss = tf.train.AdamOptimizer(learning_rate=0.0001)
    gvs = optimize_vgg_loss.compute_gradients(vgg_loss)
    gvs = [(i,j) for i,j in gvs if j in siam_vars]
    grads = [grad for grad,var in gvs]

    
    # clip the change in gradient for solving exploding gradient problem
    grads,_ = tf.clip_by_global_norm(grads,0.01)
    variables = [var for grad,var in gvs]

    
    # vgg weight optimizer
    optimizer_op = optimize_vgg_loss.apply_gradients(zip(grads,variables))


    # gate optimizer
    optimize_gate_loss = tf.train.AdamOptimizer(learning_rate=0.001).minimize(gated_loss,name='min',var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="gating"))
    
    
    sess = tf.Session()

    # TODO
    # you can rerun the trained model using the following snippet of code


    sess.run(tf.global_variables_initializer())
    new_saver = tf.train.import_meta_graph('entire_model/abcd.meta')
    new_saver.restore(sess,tf.train.latest_checkpoint('entire_model/'))
    print(sess.run(gated_loss))
    

    # saver
    saver = tf.train.Saver(max_to_keep=4,keep_checkpoint_every_n_hours=2)  
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    # fine tuning vgg weights then training soft gates and iterating over same process
    for step in range(100):

      # for i in range(100):
      #   loss_val, _ = sess.run([vgg_loss,optimizer_op])
      #   print("\n\n total avg vgg losss {} on epoch num {}  \n\n".format(loss_val/5,i))

      for i in range(100):
        loss_val, _ = sess.run([gated_loss,optimize_gate_loss])
        print("\n\n total gated losss {} on epoch num {}  \n\n".format(loss_val/5,i))

      saver.save(sess,'entire_model/abcd',global_step=step)  

# function implementing hard gating using tf.cond over the above trained model
def evaluate(model_config, train_config):

    # seeds setter
    random.seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    tf.set_random_seed(train_config['seed'])   

    g = tf.Graph()
    with g.as_default():

      # hardgated model for cross corr evaluation
      model = compAdaptiveSiam.CompSiamModel(model_config, train_config, mode='train')
      
      # builds the graph
      # should be in 0.25,0.5 or 0.75 as mentioned in paper
      model.evaluate(thresh=0.75)
      
      # model restored
      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      new_saver = tf.train.import_meta_graph('pretrained/entire_model/abcd.meta')
      new_saver.restore(sess,tf.train.latest_checkpoint('pretrained/entire_model/'))
      print("\nmodel restored\n")
      
      # computes the final cross correlation and stops when halting exceeds threshold
      cross_corr,flops,index = sess.run([model.final_cross_corr,model.final_flops,model.stop_index ])
      print("\nThe final cross corellation matrix for the image stopped at {} with {} flops".format(index,flops[0]))


if __name__ == "__main__":
  if (len(sys.argv) ==2):
    main_c,train_c,track_c = configurations()
    if sys.argv[1] =="train":
      train(main_c,train_c)
    elif sys.argv[1] =="eval":
      evaluate(main_c,train_c)

  else:
    print("Argument missing")  
