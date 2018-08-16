from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
import numpy.random as rng
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, load_model, Model
from keras.regularizers import l2
from keras.losses import binary_crossentropy

from data_provider64 import MODELS_DIR

import os
from os.path import join, dirname, isfile, abspath
from keras.preprocessing.image import img_to_array, load_img #convert image to array

np.random.seed(1337)  # for the sake of reproducibility

IMGS_DIM_2D = (64, 64)
DATA_DIR = join(dirname(dirname(__file__)), 'data')
tr_dir = join(DATA_DIR, 'train_{:d}'.format(IMGS_DIM_2D[0]))
val_dir = join(DATA_DIR, 'val_{:d}'.format(IMGS_DIM_2D[0]))
NEW_TEST_DIR = join(DATA_DIR, 'test_{:d}'.format(IMGS_DIM_2D[0]))
NEW_TEST_DIR = join(NEW_TEST_DIR, 'all')

IMGS_DIM_3D = (3, 64, 64)
CNN_MODEL_FILE = join(MODELS_DIR, 'cnn.h5')
nb_epoch = 200
BATCH_SIZE = 48
L2_REG = 0.03
W_INIT = 'he_normal'
LAST_FEATURE_MAPS_LAYER = 46
LAST_FEATURE_MAPS_SIZE = (128, 8, 8)
PENULTIMATE_LAYER = 51
PENULTIMATE_SIZE = 2048
SOFTMAX_LAYER = 55
SOFTMAX_SIZE = 475
N_author = 60




def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)



def contrastive_loss(y, d):
    """ Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    loss = K.mean(y * K.square(d) + (1 - y) * K.square(K.maximum(margin - d, 0)))
    return loss



def compute_accuracy(predictions, labels):
    """ Compute classification accuracy with a fixed threshold on distances.
    """
    return labels[predictions.ravel() < 0.5].mean()


def load_img_arr(img):
    '''Convert the path of image into a 3-D array'''
    return img_to_array(load_img(img))

def load_data(path):
    
    X = [[]for i in range(N_author)]
    i = 0 #n_author
    j = 0 #n of pictures
    for class_author in os.listdir(path):
        author_dir = abspath(join(path, class_author))

        for picture_id in os.listdir(author_dir):
            picture_path = abspath(join(author_dir, picture_id))
            X[i].append(load_img_arr(picture_path))
            j +=1
        i += 1  
    
    return np.asarray(X)


X_tr= load_data(tr_dir)

X_val= load_data(val_dir)



def create_cnn_network(imgs_dim, compile_=True):
    """ Base network to be shared (eq. to feature extraction).
    """
    cnn = Sequential()
    cnn.add(_convolutional_layer(nb_filter=16, input_shape=imgs_dim))
    cnn.add(BatchNormalization(axis=1, mode=0))
    cnn.add(PReLU(init=W_INIT))
    cnn.add(_convolutional_layer(nb_filter=16))
    cnn.add(BatchNormalization(axis=1, mode=0))
    cnn.add(PReLU(init=W_INIT))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(_convolutional_layer(nb_filter=32))
    cnn.add(BatchNormalization(axis=1, mode=0))
    cnn.add(PReLU(init=W_INIT))
    cnn.add(_convolutional_layer(nb_filter=32))
    cnn.add(BatchNormalization(axis=1, mode=0))
    cnn.add(PReLU(init=W_INIT))
    cnn.add(_convolutional_layer(nb_filter=32))
    cnn.add(BatchNormalization(axis=1, mode=0))
    cnn.add(PReLU(init=W_INIT))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    
    cnn.add(Dropout(p=0.5))

    cnn.add(Flatten())
    cnn.add(_dense_layer(output_dim=PENULTIMATE_SIZE))
    cnn.add(BatchNormalization(mode=0))
    cnn.add(PReLU(init=W_INIT))

    
    return cnn

def _convolutional_layer(nb_filter, input_shape=None):
    if input_shape:
        return _first_convolutional_layer(nb_filter, input_shape)
    else:
        return _intermediate_convolutional_layer(nb_filter)


def _first_convolutional_layer(nb_filter, input_shape):
    return Conv2D(
        nb_filter=nb_filter, nb_row=3, nb_col=3, input_shape=input_shape,
        border_mode='same', init=W_INIT, W_regularizer=l2(l=L2_REG))


def _intermediate_convolutional_layer(nb_filter):
    return Conv2D(
        nb_filter=nb_filter, nb_row=3, nb_col=3, border_mode='same',
        init=W_INIT, W_regularizer=l2(l=L2_REG))


def _dense_layer(output_dim):
    return Dense(output_dim=output_dim, W_regularizer=l2(l=L2_REG), init=W_INIT)


Xval = X_val
Xtrain = X_tr

left_input = Input(shape=(IMGS_DIM_3D))
right_input = Input(shape=(IMGS_DIM_3D))


# base_network as the shared weight network 
base_network = create_cnn_network(IMGS_DIM_3D)
processed_left = base_network(left_input)
processed_right = base_network(right_input)



distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_left, processed_right])

model = Model(input=[left_input, right_input], output=distance)



adam = Adam(lr=0.0001)
model.compile(loss=contrastive_loss, optimizer=adam)


def get_batch(n):
    """Create batch of n pairs, half same class, half different class"""
    
    #randomly sample several classes to use in the batch
    categories = rng.choice(60,size=(n,),replace=False)
    #initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((n, 3, 64, 64)) for i in range(2)]
    #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
    targets=np.zeros((n,))
    targets[n//2:] = 1

    for i in range(n):
        category = categories[i]
        idx_1 = rng.randint(0,90)
        pairs[0][i,:,:,:] = Xtrain[category,idx_1].reshape(3, 64, 64)
        idx_2 = rng.randint(0,90)
        #pick images of same class for 1st half, different for 2nd
        
        #add a random number to the category modulo n classes to ensure 2nd image has different category
        category_2 = category if i >= n//2 else (category + rng.randint(1,60)) % 60
        pairs[1][i,:,:,:] = Xtrain[category_2,idx_2].reshape(3, 64, 64)
    return pairs, targets

def make_oneshot_task(N):
        
    
    categories = rng.choice(60,size=(N,),replace=False)
    indices = rng.randint(0,10,size=(N,))
    true_category = categories[0]
    ex1, ex2 = rng.choice(10,replace=False,size=(2,))
    test_image = np.asarray([Xval[true_category,ex1]]*N).reshape(N,3, 64, 64)
    support_set = Xval[categories,indices]
    support_set[0,:,:] = Xval[true_category,ex2]
    support_set = support_set.reshape(N,3, 64, 64)
    pairs = [test_image,support_set]
    targets = np.zeros((N,))
    targets[0] = 1
    #targets, test_image, support_set = shuffle(targets, test_image, support_set)
    return pairs, targets
    
def test_oneshot(model,N,k,verbose=0):
        
    pass
    n_correct = 0
    if verbose:
        print("Evaluating model on {} unique {} way one-shot learning tasks ...".format(k,N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
    percent_correct = (100.0*n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
    return percent_correct  



evaluate_every = 500
loss_every=200
batch_size = 48
N_way = 20
n_val = 60
#model.load_weights("PATH")
best = 60.0

for i in range(2000000):
    (inputs,targets) = get_batch(batch_size)
    
    loss=model.train_on_batch(inputs,targets)
    if i % evaluate_every == 0:
        val_acc = test_oneshot(model,N_way,n_val,verbose=True)
        if val_acc >= best:
            
            best=val_acc
        print("iteration {}, Validation accuracy: {:.2f},".format(i,val_acc))

    if i % loss_every == 0:
        print("iteration {}, validation loss: {:.2f},".format(i,loss))







