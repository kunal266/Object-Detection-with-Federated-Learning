"""
Function: Preprocess data to convert to tf tensor
Input: All train data (Path to train data) from augment.py
Output:  ALL Resized image with smallest side equal to 600 px preserving original ratios


tensor_from_list = tf.convert_to_tensor(dup_train_img) # non rectangular
dataset = tf.data.Dataset.from_tensor_slices(dup_train_img)

tensor_from_list = tf.convert_to_tensor(dup_train_img_aug) # unsupported type dict
dataset = tf.data.Dataset.from_tensor_slices(dup_train_img_aug)

see https://stackoverflow.com/questions/64308232/how-to-fix-this-error-4insufficient-memory-failed-to-allocate-6220800-bytes
see https://stackoverflow.com/questions/56304986/valueerror-cant-convert-non-rectangular-python-sequence-to-tensor
for possible way to convert to tensor
data_tensor = tf.ragged.constant(data)
"""
import tensorflow as tf
import config
import augment
import cv2
# import simple_parser as sp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
C = config.Config()

# This function resize the smallest side to 600 while pretaining the original picture ratio
def get_new_img_size(width, height, img_min_side=600):
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height


def get_anchor_gt(train_data, class_count, C, img_length_calc_function):

	for img in augment.train_data:
		x_img = cv2.imread(img['filepath'])
		(width, height) = (img['width'], img['height'])
		(rows, cols, _) = x_img.shape

		assert cols == width
		assert rows == height

		# get image dimensions for resizing
		(resized_width, resized_height) = get_new_img_size(width, height, 600)

		# resize the image so that smalles side is length = 600px
		x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

		y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height,
										 img_length_calc_function)

		# resize and 0 center by mean
		x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
		x_img = x_img.astype(np.float32)
		x_img[:, :, 0] -= C.img_channel_mean[0]
		x_img[:, :, 1] -= C.img_channel_mean[1]
		x_img[:, :, 2] -= C.img_channel_mean[2]
		x_img /= C.img_scaling_factor

		x_img = np.transpose(x_img, (2, 0, 1))
		x_img = np.expand_dims(x_img, axis=0)

		y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling
		x_img = np.transpose(x_img, (0, 2, 3, 1))
		y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
		y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

		yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug


	# cv2.imwrite(img['filepath'], x_img) # writing resized image to original path










