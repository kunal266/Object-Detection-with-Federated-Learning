"""
Function: split VOC 2007 and 2012 dataset into train and validation and test (if available)
Input: path to VOC 2007 and VOC 2012 dataset
Output: list of image file name including extension .jpg of train, val, test
[filename.jpg::string]

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import numpy as np
####### VOC 2012 #######
trainfile = "data/VOCdevkit/VOC2012/ImageSets/Main/train.txt"
valfile = "data/VOCdevkit/VOC2012/ImageSets/Main/val.txt"

train_2012fname = np.loadtxt(trainfile, dtype= np.str,delimiter=' ')
val_2012fname = np.loadtxt(valfile, dtype= np.str,delimiter=' ')

print("Number of train images in VOC 2012 is", train_2012fname.shape[0])
print("Number of validation images in VOC 2012 is",val_2012fname.shape[0])

####### VOC 2007 #######
trainfile = "data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/train.txt"
valfile = "data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/val.txt"
testfile = "data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/test.txt"

#read in
train_2007fname = np.loadtxt(trainfile, dtype= np.str,delimiter=' ')
val_2007fname = np.loadtxt(valfile, dtype= np.str,delimiter=' ')
test_2007fname = np.loadtxt(testfile, dtype= np.str,delimiter=' ')
print("Number of train images in VOC 2007 is", train_2007fname.shape[0])
print("Number of validation images in VOC 2007 is",val_2007fname.shape[0])
print("Number of test images in VOC 2007 is",test_2007fname.shape[0])

# concatenate 2007 and 2012 image file name
train = list(np.concatenate([train_2007fname, train_2012fname]))
val = list(np.concatenate([val_2007fname, val_2012fname]))
test = list(test_2007fname)

# add file extension to file name
for ds in [train, val, test]:
    for i in range(len(ds)):
        ds[i] = ds[i] + '.jpg'