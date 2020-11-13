"""
Function: prepare annotations.csv file to format
[filepath,x1,y1,x2,y2,class_name]

Input: Path to annotations.csv
Output: annotate.txt
"""


import pandas as pd
# run on gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#################################
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
