"""
Function: export_annotations
Input: path to train and val xml data (not including test)
Output: write to file annotations-2012.csv and anotations-2007.csv and annotations.csv with needed annotation data
(filename, class_name, x1, y1, x2, y2)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
import pandas as pd
import xml.etree.ElementTree as ET
######### VOC 2012###########
annotations = glob.glob('data/VOCdevkit/VOC2012/Annotations/*.xml')

df = []
cnt = 0
for file in annotations:
    #filename = file.split('/')[-1].split('.')[0] + '.jpg'
    #filename = str(cnt) + '.jpg'
    filename = file.split('\\')[-1]
    filename =filename.split('.')[0] + '.jpg'
    row = []
    parsedXML = ET.parse(file)
    for node in parsedXML.getroot().iter('object'):
        objects = node.find('name').text
        xmin = int(node.find('bndbox/xmin').text)
        xmax = int(node.find('bndbox/xmax').text)
        ymin = node.find('bndbox/ymin').text # cannot convert to int because some value is float.
        ymax = int(node.find('bndbox/ymax').text)

        row = [filename, objects, xmin, xmax, ymin, ymax]
        df.append(row)
        cnt += 1

data12 = pd.DataFrame(df, columns=['filename', 'objects', 'xmin', 'xmax', 'ymin', 'ymax'])

data12[['filename', 'objects', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('annotations-2012.csv', index=False)


########voc2007 train and val##########
annotations = glob.glob('data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/*.xml')

df = []
cnt = 0
for file in annotations:
    #filename = file.split('/')[-1].split('.')[0] + '.jpg'
    #filename = str(cnt) + '.jpg'
    filename = file.split('\\')[-1]
    filename =filename.split('.')[0] + '.jpg'
    row = []
    parsedXML = ET.parse(file)
    for node in parsedXML.getroot().iter('object'):
        objects = node.find('name').text
        xmin = int(node.find('bndbox/xmin').text)
        xmax = int(node.find('bndbox/xmax').text)
        ymin = node.find('bndbox/ymin').text # cannot convert to int because some value is float.
        ymax = int(node.find('bndbox/ymax').text)

        row = [filename, objects, xmin, xmax, ymin, ymax]
        df.append(row)
        cnt += 1

data07 = pd.DataFrame(df, columns=['filename', 'objects', 'xmin', 'xmax', 'ymin', 'ymax'])

data07[['filename', 'objects', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('annotations-2007.csv', index=False)

# output file with both 07 and 12 annotations
data0712 = pd.concat([data07, data12])
data0712[['filename', 'objects', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('annotations.csv', index=False)