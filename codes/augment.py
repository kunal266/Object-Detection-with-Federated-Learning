"""
Function: augment train data with 3 choices for augment:
horizontal flips,
vertical flips,
rotation with random angle 0, 90, 180, 270
Saving augmented pictures to local

Input: Path to image data, choice of type of augmentation
Output: train_data :: list of dicts, containing all duplicated and augmented image information
For augmented data, the image file name contains a at the beginnning of original file name

"""

import cv2
import numpy as np
import copy

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import config
import random
import duplicate as dp


def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])

	if augment:
		# img_data_aug['filepath'] = img_data_aug['filepath'] + "AUG" # mark augmentation remark
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
#############DUPLICATION + AUGMENTATION###################
aug_dup_train_img= [] # list of dict
dup_train_img = [] # list of ndarray
savepath = "data/VOCtrainval-0712/JPEGImages/"
for i in range(len(dp.duplicated)):
	img_aug, img = augment(dp.duplicated[i], C, augment=True)
	cv2.imwrite(savepath + 'a' + str(i) + img_aug['filepath'].split('/')[-1], img) # save file with unique name per augmentation
	img_aug['filepath'] = savepath + 'a'  + str(i) + img_aug['filepath'].split('/')[-1]# change file path in img_aug
	aug_dup_train_img.append(img_aug)  # list of dict
	print("SAVING IMG" + img_aug['filepath'].split('/')[-1] + "AT" + savepath )
print("FINISHED AUGMENTATION WITH TRAIN DATA")


train_data = dp.duplicated.copy() + aug_dup_train_img # return list of dicts
print("NUMBER OF TRAIN IMAGES AFTER AUGMENTATION AND DUPLICATION IS",len(train_data))
##################################################
