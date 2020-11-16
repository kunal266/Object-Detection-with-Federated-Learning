"""
Function: prepare annotations.csv file to format
[filepath,x1,y1,x2,y2,class_name]

Input: Path to annotations.csv
Output:
dup-annotate.txt : Contains both duplicated train data and val data
dup-train-annotate.txt: Contains only duplicated train data
val-annotate.txt: Contains only val data
"""


import pandas as pd
import split
# run on gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
##############BEFORE DUPLICATION###################
voc = pd.read_csv("annotations.csv")
data = pd.DataFrame()
data['format'] = voc['filename']

for i in range(data.shape[0]):
    data['format'][i] = 'data/VOCtrainval-0712/JPEGImages/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(voc['xmin'][i]) + ',' + str(voc['ymin'][i]) + ','+ str(voc['xmax'][i]) + ',' + str(voc['ymax'][i]) + ',' + voc['objects'][i]

data.to_csv('annotate.txt', header=None, index=None, sep=' ')
print(data.shape)

#############AFTER DUPLICATION####################
voc = pd.read_csv("dup-annotations.csv")
data = pd.DataFrame()
data['format'] = voc['filename']

for i in range(data.shape[0]):
    data['format'][i] = 'data/VOCtrainval-0712/JPEGImages/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(voc['xmin'][i]) + ',' + str(voc['ymin'][i]) + ','+ str(voc['xmax'][i]) + ',' + str(voc['ymax'][i]) + ',' + voc['objects'][i]

data.to_csv('dup-annotate.txt', header=None, index=None, sep=' ')
print(data.shape)


# separate train and val out
dup_train = pd.DataFrame()
val = pd.DataFrame()
test = pd.DataFrame()

for i in range(len(data)):
    if data['format'][i].split("/")[-1].split(",")[0] in split.train:
        dup_train = dup_train.append(data.iloc[i])
    elif data['format'][i].split("/")[-1].split(",")[0] in split.val:
        val = val.append(data.iloc[i])
    else:
        test = test.append(data.iloc[i])

dup_train.to_csv('dup-train-annotate.txt', header=None, index=None, sep=' ')
val.to_csv('val-annotate.txt', header=None, index=None, sep=' ')

#########CHECK NUMBER OF VAL IMAGE IN SET#############
#######RERUN THIS#############







