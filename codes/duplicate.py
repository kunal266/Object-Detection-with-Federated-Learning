"""
Function: duplicate train and validation data with probability p.
means that each picture has a independent probability p of being duplicated in the training
Input:
list of image names :: str
duplicate probability p :: float
maximum number of repeat per image n :: int

Output: list of duplicated data file name with extension .jpg
Example output: [1.jpg, 1.jpg, 2,jpg, ....]
duplicate-history.csv file : recording the times each img are duplicated
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import split
import random
import numpy as np
import pandas as pd

# for reproducibility
random.seed(603008)

# set binomial parameter values
number_img = len(split.train)
max_n = 5
dup_p = 0.2

# generate binomial(n. p) for all images
x = np.random.binomial(size=number_img, n=max_n, p= dup_p)
assert len(x) == number_img, "Length of binomial is not same as number of image files"

# append duplicated data only with file name
duplicated = split.train.copy()
for i in range(number_img):
    print("Start Duplicate Image File", split.train[i], "for ", x[i], "times")
    for j in range(x[i]):
        duplicated.append(split.train[i])
print("DUPLICATION COMPLETE TOTAL NUMBER OF IMAGES AFTER DUPLICATION IS ", len(duplicated))

# create duplication record for further follow up
data = {'img': split.train, 'duplicates': x}
df = pd.DataFrame(data)
df[['img', 'duplicates']].to_csv('duplicate-history.csv', index=False)

# output dup-annotations.csv with duplicated records
working_direc = "C:/Users/Leyan/Documents/Object-Detection-with-Federated-Learning/codes"
anno = pd.read_csv(working_direc + "/annotations.csv" )

# repeat annotations by matching file names and duplicate counts
dup_anno = anno.copy() # store array
for i in range(len(df)):# df records duplicate history
    target = df.img[i] # target file name
    subset = anno[(anno.filename == target)]
    for j in range(df.duplicates[i]): # DUPLICATE FOR M TIMES
        # subset.filename = target + "DUP" + str(j)  # change subset file name
        dup_anno = dup_anno.append(subset)
    print("FINISH DUPLICATE IMAGE", target)
print("FINISH CREATING DUPLICATED ANNOTATIONS")
print("Length of Duplicated Annotations is ", len(dup_anno))
# save file
dup_anno[['filename', 'objects', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('dup-annotations.csv', index=False)


# check consistency of duplication
# c = 0
# for i in range(len(dup_anno)):
#     if (dup_anno.filename.iloc[i] in split.train):
#         c = c + 1
# AFTER DUPLICATION
# train = 46813
# val = 32182
