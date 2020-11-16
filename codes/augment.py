"""
Function: augment train data with 3 choices for augment:
horizontal flips,
vertical flips,
rotation with random angle 0, 90, 180, 270

Input: Path to image data, choice of type of augmentation
Output:
img :: list of ndarray (width, height, 3)
img_aug :: list of dict{bboxes, width, imageset, filepath, height}
QUESTIONS: DO WE NEED TO SAVE AUGMENTED PICTURES????????
"""

import cv2
import numpy as np
import copy
import simple_parser as sp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import config
import random


def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])

	if augment:
		rows, cols = img.shape[:2]

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 1)
			elif angle == 0:
				pass

			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2
				elif angle == 0:
					pass

	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]
	return img_data_aug, img


C = config.Config()
C.use_horizontal_flips = True
C.use_vertical_flips = True
C.rot_90 = True

# for reproducibility
random.seed(603008)

#############AFTER DUPLICATION###################
dup_train_img_aug = [] # list of dict
dup_train_img = [] # list of ndarray
for i in range(len(sp.dup_train_imgs)):
	img_aug, img = augment(sp.dup_train_imgs[i], C, augment=True)
	dup_train_img_aug.append(img_aug)
	dup_train_img.append(img)
print("FINISHED AUGMENTATION WITH TRAIN DATA")


############AFTER AUGMENTATION####################
train_data = sp.dup_train_imgs.copy() + dup_train_img_aug
print(len(train_data))
##################################################

