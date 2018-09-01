import cv2
import tensorflow as tf
import numpy as np
from utils.train_utils import construct_gt_score_maps
def img(val):
	cv2.imshow('image',val)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
a = {'rPos':20,'rNeg':2}
gtr = np.expand_dims(tf.Session().run(construct_gt_score_maps([15,15],5,8,a))[0],axis=-1)

gtr  = np.multiply(gtr,255)
img(gtr)
cv2.imwrite('gt15.jpg',gtr)
