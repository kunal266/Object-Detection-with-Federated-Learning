"""
Function: Convert list of dict to tensors
Input: train_data
Output:
OrderedDict :: x_img :: ndarray of img
OrderedDict :: y_img :: bboxes with class and coordinates
"""


import collections
import cv2
import pandas as pd
import tensorflow as tf

def img_to_tensor(img):
    # try to convert one img to ordered dict with tensors:
    x_img = cv2.imread(img['filepath'])
    y_img = img['bboxes']  # list of dict
    y_df = pd.DataFrame(y_img)  # df
    y_df.x1 = str(y_df.x1)  # int to str
    y_df.x2 = str(y_df.x2)
    y_df.y1 = str(y_df.y1)
    y_df.y2 = str(y_df.y2)

    x_tensor = tf.convert_to_tensor(x_img)
    y_tensor = tf.convert_to_tensor(y_df)

    mydict = collections.OrderedDict()
    mydict['x_img'] = x_tensor
    mydict['y_img'] = y_tensor

    return mydict





