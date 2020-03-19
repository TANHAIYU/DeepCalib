from __future__ import print_function

import os
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from utils_regressor_focal_dist import RotNetDataGenerator, CustomModelCheckpoint  # , angle_error
from keras import optimizers
import numpy as np
import glob, math
from shutil import copyfile
import datetime, random
import tensorflow as tf
import re
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.visible_device_list= '0, 1'
config.gpu_options.allow_growth = False
config.allow_soft_placement = False
config.log_device_placement = False
set_session(tf.Session(config=config))

model_name = 'model_multi_class/'
SAVE = "new_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
# Save
output_folder = SAVE + model_name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_log = output_folder + "Log/"
if not os.path.exists(output_log):
    os.makedirs(output_log)

output_weight = output_folder + "Best/"
if not os.path.exists(output_weight):
    os.makedirs(output_weight)

# training parameters
batch_size = 50
nb_epoch = 10000

IMAGE_FILE_PATH_DISTORTED = "/home/haiyutan/master-thesis/images/dataset/test_discrete/"

focal_start = 40
focal_end = 500
dist_end = 1.2
classes_focal = list(np.arange(focal_start, focal_end+1, 10))
classes_distortion = list(np.arange(0, 61, 1) / 50.)


def get_paths(IMAGE_FILE_PATH_DISTORTED):
    paths_train = glob.glob(IMAGE_FILE_PATH_DISTORTED + 'train/' + "*.png")
    paths_train.sort()
    parameters = []
    labels_focal_x_train = []
    labels_focal_y_train = []
    for path in paths_train:
        curr_parameter = float(re.split('_',path)[5])  #TODO
        labels_focal_x_train.append((curr_parameter - focal_start*1.) / (focal_end*1.+1. - focal_start*1.)) # normalize bewteen 0 and 1
    for path in paths_train:
        curr_parameter = float(re.split('.png', re.split('_',path)[6])[0]) #TODO
        labels_focal_y_train.append((curr_parameter - focal_start*1.) / (focal_end*1.+1. - focal_start*1.)) # normalize bewteen 0 and 1
    labels_distortion_train = []
    for path in paths_train:
        curr_parameter = float(re.split('_',path)[3]) #TODO
        labels_distortion_train.append(curr_parameter/1.2)

    c = list(zip(paths_train, labels_focal_x_train,labels_focal_y_train, labels_distortion_train))
    random.shuffle(c)
    paths_train, labels_focal_x_train,labels_focal_y_train,labels_distortion_train = zip(*c)
    paths_train, labels_focal_x_train, labels_focal_y_train,labels_distortion_train = list(paths_train), list(labels_focal_x_train), list(labels_focal_y_train),list(labels_distortion_train)
    labels_train = [list(a) for a in zip(labels_focal_x_train, labels_focal_y_train, labels_distortion_train)]

    paths_valid = glob.glob(IMAGE_FILE_PATH_DISTORTED + 'valid/' + "*.png")
    paths_valid.sort()
    parameters = []
    labels_focal_x_valid = []
    labels_focal_y_valid = []
    for path in paths_valid:
        curr_parameter = float(re.split('_',path)[5])  #TODO
        labels_focal_x_valid.append((curr_parameter-focal_start*1.)/(focal_end*1.+1.-focal_start*1.)) #normalize bewteen 0 and 1
    for path in paths_valid:
        curr_parameter = float(re.split('.png', re.split('_',path)[6])[0]) #TODO
        print(curr_parameter)
        labels_focal_y_valid.append((curr_parameter-focal_start*1.)/(focal_end*1.+1.-focal_start*1.)) #normalize bewteen 0 and 1
    labels_distortion_valid = []
    for path in paths_valid:
        curr_parameter = float(re.split('_',path)[3]) #TODO
        labels_distortion_valid.append(curr_parameter/1.2)

    c = list(zip(paths_valid, labels_focal_x_valid,labels_focal_y_valid, labels_distortion_valid))
    random.shuffle(c)
    print(c)
    paths_valid, labels_focal_x_valid, labels_focal_y_valid,labels_distortion_valid = zip(*c)
    paths_valid, labels_focal_x_valid, labels_focal_y_valid, labels_distortion_valid = list(paths_valid), list(labels_focal_x_valid), list(labels_focal_y_valid), list(labels_distortion_valid)
    labels_valid = [list(a) for a in zip(labels_focal_x_valid,labels_focal_y_valid, labels_distortion_valid)]

    return paths_train, labels_train, paths_valid, labels_valid


paths_train, labels_train, paths_valid, labels_valid = get_paths(IMAGE_FILE_PATH_DISTORTED)

print(len(paths_train), 'train samples')
print(len(paths_valid), 'valid samples')

input_shape = (310, 1024, 3)   #default:(299, 299, 3)
main_input = Input(shape=input_shape, dtype='float32', name='main_input')
phi_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=main_input, input_shape=input_shape)
phi_features = phi_model.output
phi_flattened = Flatten(name='phi-flattened')(phi_features)
final_output_focal_x = Dense(1, activation='sigmoid', name='output_focal_x')(phi_flattened)
final_output_focal_y = Dense(1, activation='sigmoid', name='output_focal_y')(phi_flattened)
final_output_distortion = Dense(1, activation='sigmoid', name='output_distortion')(phi_flattened)

layer_index = 0
for layer in phi_model.layers:
    layer.name = layer.name + "_phi"

model = Model(input=main_input, output=[final_output_focal_x,final_output_focal_y,final_output_distortion])

learning_rate = 10 ** -6

adam = optimizers.Adam(lr=learning_rate)
model.compile(loss={'output_focal_x':'logcosh','output_focal_y':'logcosh', 'output_distortion':'logcosh'},
              optimizer=adam,
              metrics={'output_focal_x':'logcosh','output_focal_y':'logcosh','output_distortion':'logcosh'} )
model.summary()
model_json = phi_model.to_json()

with open(output_folder + "model.json", "w") as json_file:
    json_file.write(model_json)

copyfile(os.path.basename(__file__), output_folder + os.path.basename(__file__))

tensorboard = TensorBoard(log_dir=output_log)

checkpointer = CustomModelCheckpoint(
    model_for_saving=model,
    filepath=output_weight + "weights_{epoch:02d}_{val_loss:.2f}.h5",
    save_best_only=True,
    monitor='val_loss',
    save_weights_only=True
)

generator_training = RotNetDataGenerator(input_shape=input_shape, batch_size=batch_size, one_hot=True,
                                         preprocess_func=preprocess_input, shuffle=True).generate(paths_train,
                                                                                                  labels_train)
generator_valid = RotNetDataGenerator(input_shape=input_shape, batch_size=batch_size, one_hot=True,
                                      preprocess_func=preprocess_input, shuffle=True).generate(paths_valid,
                                                                                               labels_valid)

# training loop
model.fit_generator(
    generator=generator_training,
    steps_per_epoch=(len(paths_train) // batch_size), # 29977
    epochs=nb_epoch,
    validation_data=generator_valid,
    validation_steps=(len(paths_valid) // batch_size),
    callbacks=[tensorboard, checkpointer],
    use_multiprocessing=True,
    workers=2,
    #verbose=3
)
