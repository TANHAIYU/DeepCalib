# this is for images from the test dir, better name would be "evaluate"
from __future__ import print_function

import os, cv2, sys
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras import optimizers
import numpy as np
import glob, os, pdb
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.allow_soft_placement = True
# set_session(tf.Session(config=config))

results_path = "/media/cml-kaist/My Passport/8 - data/new_fisheye_images/results/"
IMAGE_FILE_PATH_DISTORTED = "/home/cml-kaist/Desktop/Dohyeong/Undistortion/Try1/frames/"

path_to_weights = '/media/cml-kaist/My Passport/3 - prediction/Classification/Single_net/weights_06_5.61.h5'

filename_results = results_path + 'airport.txt'

#if os.path.exists(filename_results):
   # sys.exit("file exists")

classes_focal = list(np.arange(40, 501, 10))  # focal
classes_distortion = list(np.arange(0, 61, 1) / 50.)


def get_paths(IMAGE_FILE_PATH_DISTORTED):
    paths_test = glob.glob(IMAGE_FILE_PATH_DISTORTED + "*.jpg")
    paths_test.sort()

    return paths_test


paths_test = get_paths(IMAGE_FILE_PATH_DISTORTED)

print(len(paths_test), 'test samples')

#for image in paths_test:
#    print(image)

# pdb.set_trace()


with tf.device('/gpu:0'):
    input_shape = (299, 299, 3)
    main_input = Input(shape=input_shape, dtype='float32', name='main_input')
    phi_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=main_input, input_shape=input_shape)
    phi_features = phi_model.output
    phi_flattened = Flatten(name='phi-flattened')(phi_features)
    final_output_focal = Dense(len(classes_focal), activation='softmax', name='output_focal')(phi_flattened)
    final_output_distortion = Dense(len(classes_distortion), activation='softmax', name='output_distortion')(
        phi_flattened)

    layer_index = 0
    for layer in phi_model.layers:
        layer.name = layer.name + "_phi"

    model = Model(input=main_input, output=[final_output_focal, final_output_distortion])
    model.load_weights(path_to_weights)

    n_acc_focal = 0
    n_acc_dist = 0
    print(len(paths_test))

    file = open(filename_results, 'a')
    # for i, path in enumerate(paths_test):
    #     if i % 1000 == 0:
    #         print(i, ' ', len(paths_test))
    image = cv2.imread('/media/cml-kaist/My Passport/8 - data/new_fisheye_images/church.jpg')
    image = cv2.resize(image, (299, 299))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.

    image = image - 0.5

    image = image * 2.
    image = np.expand_dims(image, 0)

    image = preprocess_input(image)  ### sehr wichtig, brauchen wir das #########

    # loop
    prediction_focal = model.predict(image)[0]
    prediction_dist = model.predict(image)[1]
    n_acc_focal += classes_focal[np.argmax(prediction_focal[0])]
    n_acc_dist += classes_distortion[np.argmax(prediction_dist[0])]

    #file.write(path + '\tprediction_focal\t' + str(
        #classes_focal[np.argmax(prediction_focal[0])]) + '\tprediction_dist\t' + str(
       # classes_distortion[np.argmax(prediction_dist[0])]) + '\n')
    # print(' ')
    print('focal:')
    print(classes_focal[np.argmax(prediction_focal[0])])

    print('dist:')
    print(classes_distortion[np.argmax(prediction_dist[0])])