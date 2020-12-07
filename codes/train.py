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

# create classes count of train imgs
classes_count = {}
for img in train_imgs:
    for box in img['bboxes']:
        class_name = box['class']
        if class_name not in classes_count:
            classes_count[class_name] = 1
        else:
            classes_count[class_name] += 1

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

assert K.common.image_dim_ordering() == 'tf', "Wrong image ordering dimension, Please change img input shape to (3, None, None)"
input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
from codes import vgg as nn
shared_layers = nn.nn_base(img_input, trainable=True)

# pass the settings from the command line, and persist them in the config object
C = config.Config()

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

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={f'dense_class_{len(classes_count)}': 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')


epoch_length = 50
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))

rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

# Next STEP: try to produce class_mapping for final data from the sp.py
class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True


