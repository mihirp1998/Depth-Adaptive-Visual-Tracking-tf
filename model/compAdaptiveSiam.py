from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import vgg19_trainable as vgg19

import functools
import numpy as np
import tensorflow as tf

from datasets.dataloader import DataLoader
from metrics.track_metrics import center_dist_error, center_score_error
from utils.train_utils import construct_gt_score_maps, load_mat_model
from scipy.stats import kurtosis,entropy
slim = tf.contrib.slim


class CompSiamModel:
	def __init__(self, model_config, train_config, mode='train'):
		self.model_config = model_config
		self.train_config = train_config
		self.mode = mode
		assert mode in ['train', 'validation', 'inference']

		if self.mode == 'train':
			self.data_config = self.train_config['train_data_config']
		elif self.mode == 'validation':
			self.data_config = self.train_config['validation_data_config']

		self.dataloader = None
		self.keyFrame = None
		self.searchFrame = None
		self.response = None
		self.batch_loss = None
		self.total_loss = None
		self.init_fn = None
		self.global_step = None

	def is_training(self):
		return self.mode == 'train'



		# code used to get the data from pickled files represents the first frame and the next fram
	def build_inputs(self):
		if self.mode in ['train', 'validation']:
			with tf.device("/cpu:0"):  # Put data loading and preprocessing in CPU is substantially faster
				self.dataloader = DataLoader(self.data_config, self.is_training())
				self.dataloader.build()
				keyFrame, searchFrame = self.dataloader.get_one_batch()
				keyFrame = tf.to_float(keyFrame)
				searchFrame = tf.to_float(searchFrame)
		else:
			self.examplar_feed = tf.placeholder(shape=[None, None, None, 3],
																					dtype=tf.uint8,
																					name='examplar_input')
			self.searchFrame_feed = tf.placeholder(shape=[None, None, None, 3],
																					dtype=tf.uint8,
																					name='searchFrame_input')
			keyFrame = tf.to_float(self.examplar_feed)
			searchFrame = tf.to_float(self.searchFrame_feed)

			# images are rescaled to solve the exploding gradient problem *NOT SUGGESTED IN PAPER*
		self.keyFrame = keyFrame/128
		self.searchFrame = searchFrame/128



		# code for creating seperate vgg networks for keyframe and search frame
	def build_image_nets(self,vgg_pretrain= None, reuse=False):
		
		config = self.model_config['embed_config']
  
		sess = tf.Session()
		
		# key frame network 
		vgg_keyFrame = vgg19.Vgg19(vgg_pretrain)
		vgg_keyFrame.build(self.keyFrame)


		# search frame network
		vgg_searchFrame = vgg19.Vgg19(vgg_pretrain)
		vgg_searchFrame.build(self.searchFrame)

		self.keyFrame_net =  vgg_keyFrame
		self.searchFrame_net =  vgg_searchFrame



		# code for creating the cross correlation map for the two images
	def build_detection(self, curr_searchFrame_embeds, curr_templates, reuse=False):
		with tf.variable_scope('detection', reuse=tf.AUTO_REUSE):
			def _translation_match(x, z):  # translation match for one example within a batch
				x = tf.expand_dims(x, 0)  # [1, in_height, in_width, in_channels]
				z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, 1]
				return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')



			output = tf.map_fn(lambda x: _translation_match(x[0], x[1]),
												 (curr_searchFrame_embeds, curr_templates),
												 dtype=curr_searchFrame_embeds.dtype)

			output = tf.squeeze(output, [1, 4])  # of shape e.g., [8, 15, 15]


			# Adjust score, this is required to make training possible.
			config = self.model_config['adjust_response_config']
			bias = tf.get_variable('biases', [1],
														 dtype=tf.float32,
														 initializer=tf.constant_initializer(0.0, dtype=tf.float32),
														 trainable=config['train_bias'])
			response = config['scale'] * output + bias

			# response refers to the cross correlation map between two images
			return response

			

			# building 5 blocks, flops and cross-corr maps for each image as mentioned in the paper
	def build_blocks(self):

		keyFrame_net = self.keyFrame_net
		searchFrame_net = self.searchFrame_net

		# block 1
		self.block1_keyFrame_embed = keyFrame_net.pool1
		self.block1_searchFrame_embed = searchFrame_net.pool1
		block1_flops = 2 * searchFrame_net.flops1


		block1_cross_corr = self.build_detection(self.block1_searchFrame_embed, self.block1_keyFrame_embed,reuse=True)

		# block 2
		block2_keyFrame_embed = keyFrame_net.pool2
		block2_searchFrame_embed = searchFrame_net.pool2
		block2_flops = 2 * searchFrame_net.flops2

	 
		block2_cross_corr = self.build_detection(block2_searchFrame_embed, block2_keyFrame_embed, reuse=False)


		# block 3
		block3_keyFrame_embed = keyFrame_net.pool3
		block3_searchFrame_embed = searchFrame_net.pool3
		block3_flops = 2 * searchFrame_net.flops3


		block3_cross_corr = self.build_detection(block3_searchFrame_embed, block3_keyFrame_embed,reuse=False)


		# block 4
		block4_keyFrame_embed = keyFrame_net.pool4
		block4_searchFrame_embed = searchFrame_net.pool4
		block4_flops = 2 * searchFrame_net.flops4


		block4_cross_corr = self.build_detection(block4_searchFrame_embed, block4_keyFrame_embed,reuse=False)


		# block 5
		block5_keyFrame_embed = keyFrame_net.pool5
		block5_searchFrame_embed = searchFrame_net.pool5
		block5_flops = 2 * searchFrame_net.flops5


		block5_cross_corr = self.build_detection(block5_searchFrame_embed, block5_keyFrame_embed,reuse=True)


		# number of flops for each block in vgg net
		self.flops_metric =  [block1_flops,block2_flops,block3_flops,block4_flops,block5_flops]

		# cross correlation maps for each block
		self.cross_corr = [block1_cross_corr, block2_cross_corr, block3_cross_corr, block4_cross_corr, block5_cross_corr]



		# code used to create ground truth box intersection between two neightbouring image
	def block_loss(self, block_cross_corr):

		cross_corr_size = block_cross_corr.get_shape().as_list()[1:3]  # [height, width]
		print("the batch size ",self.data_config['batch_size'])

		# ground truth box
		gt = construct_gt_score_maps(cross_corr_size,self.data_config['batch_size'],
																 self.model_config['embed_config']['stride'],
																 self.train_config['gt_config'])

		
		with tf.name_scope('Loss'):
				# softmax cross entropy used to measure loss as mentioned in paper
			loss = tf.losses.softmax_cross_entropy(gt,block_cross_corr)
		return loss


		# shallow feature extractor inorder to save computation time... Non Differentiable

		# INot sure if I have implemented each part of it in the right way
	def shallow_feature_extractor(self):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			cross_corr = sess.run(self.cross_corr)
			# flattened the cross corr for calculating kurtosis and entropy

			cross_corr_1d = [np.reshape(i,[i.shape[0],-1]) for i in cross_corr]
			# no tensorflow function found for finding kurtosis and entropy so computed it
			
			kurtosis_val = [kurtosis(i,axis=1) for i in cross_corr_1d]
			entropy_val = [entropy(i.T) for i in cross_corr_1d]

 			# expand dimension
 			kurtosis_val = np.expand_dims(np.array(kurtosis_val),axis=2)
			entropy_val  =  np.expand_dims(np.array(entropy_val),axis=2)

			# first five values of cross corr
			first_five  = np.array([i[:,:5]for i in cross_corr_1d])

			# max five values
			max_five = np.array([np.sort(i,axis = 1)[:,::-1][:,:5] for i in cross_corr_1d])
			
			shallow_features = np.concatenate([kurtosis_val,entropy_val,first_five,max_five],axis=2)
			self.shallow_features = tf.Variable(shallow_features,trainable=False)
			# print(max_five[0,0,:],shallow_features.shape)
	

	# function for adaptive computation with early stopping FOR hard gating while evaluation
	def act_early_stop(self,thresh= 0.5):
		# run this function when computation is stopped
		def same(i):
			curr_cross = self.cross_corr[i-1]
			curr_shallow = self.shallow_features[i-1]
			curr_budgetedscore = self.gStarFunc(i-1)
			curr_flops = self.flops_metric[i-1]
			key = i
			return curr_cross,curr_shallow,curr_flops,curr_budgetedscore,key

		# run this function when the budgeted confidence score is below the threshold 
		def next(i):
			curr_cross = self.cross_corr[i]
			curr_shallow = self.shallow_features[i]
			curr_budgetedscore = self.gStarFunc(i)
			curr_flops = self.flops_metric[i-1]
			key = i
			key += 1
			return curr_cross,curr_shallow,curr_flops,curr_budgetedscore,key	
		key = 0	
		# run the early stopping
		for i in range(5):
			if i ==0:
				return_values =  next(key)
			else:
				# the main func for early stop
				return_values =  tf.cond(finished_batch,lambda: same(i), lambda: next(i))
			curr_cross,curr_shallow,curr_flops,curr_budgetedscore,key = return_values
			# boolean for stopping
			finished = curr_budgetedscore > thresh
			# and over booleans
			finished_batch = tf.reduce_all(finished)
		
		# final values
		final_cross_corr = curr_cross
		final_flops = curr_flops
		return final_cross_corr,final_flops,key-1	




		# the g function formula which computes sigmoid 
	def gFunction(self):
		with tf.variable_scope('gating', reuse=False):
			self.gFuncResults =  tf.layers.dense(self.shallow_features,1,activation=tf.sigmoid)

		# Budgeted Gating Function 
	def gStarFunc(self,i):
		if i < 4:
			gStarSum = 0
			for j in range(i):
				gStarSum = gStarSum + self.gStarFunc(j)
			gStarValue = (1 - gStarSum) *self.gFuncResults[i] 	
			return gStarValue
		elif i == 4:
			gStarSum = 0
			for i in range(4):
				gStarSum = gStarSum + self.gStarFunc(i)
			return gStarSum
		else:
			return 0		 

		# Gate loss formula in paper
	def gateLoss(self,lamda=0.5):
		total_gate_loss = 0 
			# table values for incremental additional cost as mentioned in paper
		p_table = [1,1.43,3.35,3.4,0.95]
		tracking_loss = 0
		computational_loss = 0
		for i in range(5):
			gStarVal = self.gStarFunc(i)
			
			# tracking loss
			tracking_loss += gStarVal* self.block_losses[i]
			
			# computation loss
			computational_loss += p_table[i]* gStarVal
		
		# lamda ratio between track loss and comp loss
		total_gate_loss = tracking_loss + lamda*computational_loss	
		self.total_gate_loss = tf.reduce_mean(total_gate_loss)
		print(self.total_gate_loss)	

		# Intermediate Supervision loss for all blocks .. Also causes Exploding gradient
	def build_block_loss(self):

		cross_corr_arr = self.cross_corr
		loss = None
		self.block_losses = [self.block_loss(i) for i in cross_corr_arr]
		# total loss
		self.total_loss = tf.losses.get_total_loss()
						

		# function for evaluation which incluedes act
	def evaluate(self,vgg_pretrain= None,thresh=0.5):
		with tf.name_scope("validate"):
			self.build_inputs()
			self.build_image_nets(reuse=True)
			self.build_blocks()
			self.shallow_feature_extractor()
			self.gFunction()
			self.final_cross_corr,self.final_flops,self.stop_index  = self.act_early_stop(thresh = thresh)

			# training function
	def build(self, reuse=False,vgg_pretrain= None):
		with tf.name_scope(self.mode):
			self.build_inputs()
			self.build_image_nets(reuse=reuse,vgg_pretrain= vgg_pretrain)
			self.build_blocks()
			self.shallow_feature_extractor()
			self.gFunction()
			self.build_block_loss()
			self.gateLoss()
			print("done")
