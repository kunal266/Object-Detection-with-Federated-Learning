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
def filter_client(x):
  return tf.math.equal(x['id'], 1)

# dataset = all_client_ds.filter(filter_client)

# try to extract labels back from tf.dataset
# https://stackoverflow.com/questions/56226621/how-to-extract-data-labels-back-from-tensorflow-dataset


# Way2: define a tf.dataset for whole population from generator then use dataset.filter function
# def generator return dataset for each client
temp = tf.data.Dataset.from_generator(fedGen,
                                      (tf.uint8, tf.string, tf.int32),
                                      ((None, None, 3), (None, 5), ()),
                                      args = data_gen_train) # not sure if this is correct

list(temp.take(3).as_numpy_iterator())

# Way3: Filter all elements by buit in functions
a = filter(lambda x : list(x)[0]['id'] < 5 , all_client_ds)
















