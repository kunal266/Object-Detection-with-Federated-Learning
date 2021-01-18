import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from codes.simple_parser import get_data
from codes import config, data_generators
from codes.augment import train_data
# from codes.vgg import get_img_output_length
from codes.simple_parser import val_imgs

# vgg not imported yet
# import vgg.get_img_output_length
def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)

# pass the settings from the command line, and persist them in the config object
C = config.Config()

train_imgs = train_data # duplicated and augmented train data

# create classes count and class mapping of train imgs
classes_count = {}
class_mapping = {}
found_bg = False
for img in train_imgs:
    for box in img['bboxes']:
        class_name = box['class']
        if class_name not in classes_count:
            classes_count[class_name] = 1
        else:
            classes_count[class_name] += 1

        if class_name not in class_mapping:
            if class_name == 'bg' and found_bg == False:
                print(
                    'Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                found_bg = True
            class_mapping[class_name] = len(class_mapping)
if found_bg:
    if class_mapping['bg'] != len(class_mapping) - 1:
        key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
        val_to_switch = class_mapping['bg']
        class_mapping['bg'] = len(class_mapping) - 1
        class_mapping[key_to_switch] = val_to_switch
# NOTE: Currently class bg is not in class_mapping, we add in bg in last of the mapping for debugging purpose
if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)


import random
random.shuffle(train_imgs)
num_imgs = len(train_imgs)

print(f'Num train samples {len(train_imgs)}')
print(f'Num val samples {len(val_imgs)}')

# try to assign 100 client ID to all train pictures
# this is iid setting where each client share similar number of images following uniform distribution
random.seed(603008)
imgs_cid = [] # list contain all cid for imgs
for i in range(len(train_imgs)):
    imgs_cid.append(random.randint(0, 100))
# append id attribute to dict
j = 0
for img in train_imgs:
    img['id'] = imgs_cid[j]
    j = j + 1
# make sure index is right
assert j == num_imgs, "Number of train images not consistent with number of assigned client id"




# Prelim analysis
# Number of img per client
from collections import Counter
print(Counter(imgs_cid))

# create data generators returning model input

data_gen_train = data_generators.get_anchor_gt(train_imgs,  C, get_img_output_length,  mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, C, get_img_output_length, mode='val')

# Not sure why rpn output is all 0?????

from codes import img_to_tensor
all_client_ds = img_to_tensor.fedGen(data_gen_train) # this gives tf.dataset output
all_client_ds_v2 = img_to_tensor.fedGen_v2(data_gen_train) # this gives ordered dict output, need to further transform to tf.ds
example_dataset = next(all_client_ds) # example element of all client ds
example_dataset_v2 = next(all_client_ds_v2)
# define a function returning client dataset for specified data
# cost too much time !!!!!!!!!!!!!!!!!!
client_ids = list(range(0, 100))
# c = tf.data.Dataset.from_tensor_slices([a, b])
import tensorflow as tf
# Way1: Iterate over all elements and filter by hand
# @tf.function  # to speed up codes
# def create_ds_for_clients(id):
#     assert id in client_ids, "No data found for the specified client"
#     client_element = [] # list of all
#     for each in all_client_ds: # loop over all data
#         e = list(each)[0] # extract element
#         if e['id'] == id: # find corresponding data
#             client_element.append(each)
#         return tf.data.Dataset.from_tensor_slices(client_element)
#
# ds = create_ds_for_clients(0)


# `tf.math.equal(x, y)` is required for equality comparison
# def filter_client(x):
#   return tf.math.equal(x['id'], 1)

# dataset = all_client_ds.filter(filter_client)

# try to extract labels back from tf.dataset
# https://stackoverflow.com/questions/56226621/how-to-extract-data-labels-back-from-tensorflow-dataset


# Way2: define a tf.dataset for whole population from generator then use dataset.filter function
# def generator return dataset for each client
# temp = tf.data.Dataset.from_generator(img_to_tensor.fedGen,
#                                       ({'x_img': tf.uint8, 'y_img': tf.string, 'id' :tf.int32}), # should be contained in tensor
#                                       ({'x_img': tf.TensorShape((None, None, 3)),
#                                         'y_img': tf.TensorShape((None, 5)),
#                                         'id':tf.TensorShape(None)}),
#                                       args = next(data_gen_train)) # not sure if this is correct
#
# list(temp.take(3).as_numpy_iterator())
# Problem 1: args cannot be generator, must be sequence
# Problem 2:  Can't convert non-rectangular Python sequence to Tensor. --> Ragged tensor
# Problem 3: As with normal Tensors, the values in a RaggedTensor must all have the same type;
# and the values must all be at the same nesting depth (the rank of the tensor):

# Way3: Filter all elements by built in functions
# a = filter(lambda x : list(x)[0]['id'] < 5 , all_client_ds)


# Way 4: Create dict collecting all img info, then only yield data for the current ID in the generator
# Create dict keyed by id
client_img = {}
for img in train_imgs:
    id = img['id']
    if str(id) not in client_img.keys():
        client_img[str(id)] = [img]
    else:
        client_img[str(id)].append(img)

def create_ds_for_client(id):
    assert id in client_ids, "No data found for the specified client"
    client_train_imgs = client_img[str(id)]
    client_data_gen_train = data_generators.get_anchor_gt(client_train_imgs, C, get_img_output_length, mode='train')
    client_ds = img_to_tensor.fedGen(client_data_gen_train)
    return client_ds

# Next step: Create federated dataset <- All client ids + create_ds_for_client()
gen = create_ds_for_client(16)
# temp = tf.data.Dataset.from_generator(gen,
#                                       ({'x_img': tf.uint8, 'y_img': tf.string, 'id' :tf.int32}), # should be contained in tensor
#                                       ({'x_img': tf.TensorShape((None, None, 3)),
#                                         'y_img': tf.TensorShape((None, 5)),
#                                         'id':tf.TensorShape(None)}))

# Current problem: generator -> tf.dataset or -> list of client data



# Model part:
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
K.set_learning_phase(True)
assert K.common.image_dim_ordering() == 'tf', "Wrong image ordering dimension, Please change img input shape to (3, None, None)"
input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
from codes import vgg as nn
shared_layers = nn.nn_base(img_input, trainable=True)

# pass the settings from the command line, and persist them in the config object
import re
C = config.Config()
# specify model save weights
C.model_path = "frcnn_model_weights.hdf5"
model_path_regex = re.match("^(.+)(\.hdf5)$", C.model_path)
if model_path_regex.group(2) != '.hdf5':
	print('Output weights must have .hdf5 filetype')
	exit(1)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

# get pretrained weights
C.base_net_weights = nn.get_weight_path()
try:
	print('loading weights from {C.base_net_weights}')
	model_rpn.load_weights(C.base_net_weights, by_name=True)
	model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')


from codes import losses as losses
import numpy as np
import time
import codes.roi_helpers as roi_helpers
from keras.utils import generic_utils
# Loss and optimizer
optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={f'dense_class_{len(classes_count)}': 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

# Initialize training settings
epoch_length = 50
num_epochs = int(10)
iter_num = 0

losses = np.zeros((epoch_length, 5))

rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf


class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True


# Training loop:

for epoch_num in range(num_epochs):

    progbar = generic_utils.Progbar(epoch_length)
    print(f'Epoch {epoch_num + 1}/{num_epochs}')

    while True:
        try:

            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print(
                    f'Average number of overlapping bounding boxes from RPN = {mean_overlapping_bboxes} for {epoch_length} previous iterations')
                if mean_overlapping_bboxes == 0:
                    print(
                        'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            X, Y, img_data, _ = next(data_gen_train)

            loss_rpn = model_rpn.train_on_batch(X, Y)

            P_rpn = model_rpn.predict_on_batch(X)

            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.common.image_dim_ordering(), use_regr=True,
                                       overlap_thresh=0.7, max_boxes=300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois // 2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=True).tolist()

                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],  [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
            # Raise error : Invalid argument: You must feed a value for placeholder tensor 'time_distributed_3/keras_learning_phase' with dtype bool
            # Possible solution see https://github.com/keras-team/keras/issues/5940
            # K.set_learning_phase(1) before compiling the model do not work?

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            progbar.update(iter_num + 1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                          ('detector_cls', losses[iter_num, 2]),
                                          ('detector_regr', losses[iter_num, 3])])

            iter_num += 1

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print(
                        f'Mean number of bounding boxes from RPN overlapping ground truth boxes: {mean_overlapping_bboxes}')
                    print(f'Classifier accuracy for bounding boxes from RPN: {class_acc}')
                    print(f'Loss RPN classifier: {loss_rpn_cls}')
                    print(f'Loss RPN regression: {loss_rpn_regr}')
                    print(f'Loss Detector classifier: {loss_class_cls}')
                    print(f'Loss Detector regression: {loss_class_regr}')
                    print(f'Elapsed time: {time.time() - start_time}')

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print(f'Total loss decreased from {best_loss} to {curr_loss}, saving weights')
                    best_loss = curr_loss
                model_all.save_weights(
                    model_path_regex.group(1) + "_" + '{:04d}'.format(epoch_num) + model_path_regex.group(2))

                break

        except Exception as e:
            print(f'Exception: {e}')
            continue

print('Training complete, exiting.')


# start write tff part:

example_dataset = create_ds_for_client(16) # gives a generator
example_element = next(iter(example_dataset)) # gives 1 img input from the client ds

NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

def preprocess(dataset):
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
      BATCH_SIZE).prefetch(PREFETCH_BUFFER)

preprocessed_example_dataset = preprocess(example_dataset)
sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(preprocessed_example_dataset)))

sample_batch


def create_keras_model():
  return model_all

def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
