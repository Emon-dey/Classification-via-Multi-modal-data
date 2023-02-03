import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image

from keras.models import Sequential,Model,load_model
from keras.layers import LeakyReLU,Input, Dense, Activation, Dropout, LSTM, Flatten, Embedding, merge,TimeDistributed,Multiply,Concatenate,Bidirectional,BatchNormalization,Add,AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Reshape,Permute,Lambda,multiply
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import metrics
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

# from keras_bert import get_custom_objects
# from tqdm import tqdm
# from chardet import detect
# import keras
# # from keras_bert import load_trained_model_from_checkpoint
# # from keras_bert import Tokenizer as k_Tokenizer

import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer
# from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix,auc
from sklearn.utils import class_weight
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
#from skimage.color import rgb2gray

import nltk

import os, argparse,cv2,h5py,string
import collections
from collections import Counter


from scipy import interp
from itertools import cycle
import seaborn as sns

# from deepexplain.tensorflow import DeepExplain
# import codecs

# import innvestigate
# import innvestigate.utils as iutils

# from skimage.measure import compare_ssim as ssim
# from skimage import data
# from skimage.color import rgb2gray

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# from tensorflow.compat.v1 import ConfigProto,InteractiveSession
# from keras.backend.tensorflow_backend import set_session,clear_session,get_session



# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1
# set_session(tf.Session(config=config))

# config = tf.ConfigProto()
# config.gpu_options.visible_device_list = "1"
# config.gpu_options.per_process_gpu_memory_fraction = .3
# set_session(tf.Session(config=config))

# G =tf.Graph()
# sess1 = tf.Session(graph=G, config=tf.ConfigProto(log_device_placement=False,gpu_options=tf.GPUOptions(allow_growth=True,visible_device_list='0')))
# sess2 = tf.Session(graph=G, config=tf.ConfigProto(log_device_placement=False,gpu_options=tf.GPUOptions(allow_growth=True,visible_device_list='1')))



# Seed value

# Apparently you may use different seed values at each stage
seed_value= 443512

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
#tf.set_random_seed(seed_value)

#tf.random.set_seed()




