"""
Function: A generator returning model input of all train imgs
Input: data_gen_train
Output:
OrderedDict :: x_img :: ndarray of img
OrderedDict :: y_img :: bboxes with class and coordinates
OrderedDict :: id :: client id for the current image
"""


import collections
import cv2
import pandas as pd
import tensorflow as tf

def fedGen(data_gen_train):
    for each in data_gen_train:
        img = each[-2]
        x_img = cv2.imread(img['filepath'])
        y_img = img['bboxes']  # list of dict
        y_df = pd.DataFrame(y_img)  # df
        y_df.x1 = str(y_df.x1)  # int to str
        y_df.x2 = str(y_df.x2)
        y_df.y1 = str(y_df.y1)
        y_df.y2 = str(y_df.y2)

        x_tensor = tf.convert_to_tensor(x_img)
        y_tensor = tf.convert_to_tensor(y_df)
        id = tf.convert_to_tensor(each[-1]) # last element returned

        mydict = collections.OrderedDict()
        mydict['x_img'] = x_tensor
        mydict['y_img'] = y_tensor
        mydict['id'] = id
        # convert ordered dict to tensor
        myts = tf.data.Dataset.from_tensors(mydict)
        yield myts
        # use tf.data.Dataset.from_generator?
        # from_generator(
        #     generator, output_types, output_shapes=None, args=None
        # )

def fedGen_v2(data_gen_train):
    for each in data_gen_train:
        img = each[-2]
        x_img = cv2.imread(img['filepath'])
        y_img = img['bboxes']  # list of dict
        y_df = pd.DataFrame(y_img)  # df
        y_df.x1 = str(y_df.x1)  # int to str
        y_df.x2 = str(y_df.x2)
        y_df.y1 = str(y_df.y1)
        y_df.y2 = str(y_df.y2)

        x_tensor = tf.convert_to_tensor(x_img)
        y_tensor = tf.convert_to_tensor(y_df)
        id = tf.convert_to_tensor(each[-1]) # last element returned

        mydict = collections.OrderedDict()
        mydict['x_img'] = x_tensor
        mydict['y_img'] = y_tensor
        mydict['id'] = id
        # convert ordered dict to tensor
        # myts = tf.data.Dataset.from_tensors(mydict)
        yield mydict

def fedGen_v3(data_gen_train):
    all_dict = {}
    for each in data_gen_train:
        img = each[-2]
        x_img = cv2.imread(img['filepath'])
        y_img = img['bboxes']  # list of dict
        y_df = pd.DataFrame(y_img)  # df
        y_df.x1 = str(y_df.x1)  # int to str
        y_df.x2 = str(y_df.x2)
        y_df.y1 = str(y_df.y1)
        y_df.y2 = str(y_df.y2)

        x_tensor = tf.convert_to_tensor(x_img)
        y_tensor = tf.convert_to_tensor(y_df)
        id = tf.convert_to_tensor(each[-1]) # last element returned

        mydict = collections.OrderedDict()
        mydict['x_img'] = x_tensor
        mydict['y_img'] = y_tensor
        # mydict['id'] = id

        all_dict[str(id)] = mydict
        # convert ordered dict to tensor
        # myts = tf.data.Dataset.from_tensors(mydict)
    return all_dict

