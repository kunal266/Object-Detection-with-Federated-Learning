"""
Function: plots
Input: path to image data
Output: plot boxes for each image form the dataset
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from copy import copy


def filterFiles(directoryPath, extension):
    """
        This function filters the format files with the selected extension in the directory

        Args:
            directoryPath (str): relative path of the directory that contains text files
            extension (str): extension file

        Returns:
            The list of filtered files with the selected extension
    """
    relevant_path = directoryPath
    included_extensions = [extension]
    file_names = [file1 for file1 in os.listdir(relevant_path) if
                  any(file1.endswith(ext) for ext in included_extensions)]
    numberOfFiles = len(file_names)
    listParams = [file_names, numberOfFiles]
    return listParams

[image_names, numberOfFiles] = filterFiles("data/VOCdevkit/VOC2012/JPEGImages", "jpg")

print("Total number of images is ", numberOfFiles)

trainRCNN = pd.read_csv('annotations.csv', sep=",", header=None)
trainRCNN.columns = ['filename', 'objects', 'xmin', 'xmax', 'ymin', 'ymax']
trainRCNN.head()

for imageFileName in image_names[0:5]: # plot first 5 images
    fig = plt.figure()
    # add axes to the image
    ax = fig.add_axes([0, 0, 1, 1])  # adding X and Y axes from 0 to 1 for each direction
    plt.axis('off')

    # read and plot the image
    image = plt.imread('data/VOCdevkit/VOC2012/JPEGImages/' + imageFileName)
    plt.imshow(image)
    # iterating over the image for different objects
    for _, row in trainRCNN[trainRCNN.filename == imageFileName].iterrows():
        xmin = float(row.xmin)
        xmax = float(row.xmax)
        ymin = float(row.ymin)
        ymax = float(row.ymax)

        width = xmax - xmin
        height = ymax - ymin
        ClassName = row.objects
        # assign different color to different classes of objects
        if row.objects == 'person':
            ax.annotate('person', xy=(xmax - 40, ymin + 20))
            rect = patches.Rectangle((xmin, ymin), width, height, edgecolor='r', facecolor='none')
        elif row.objects == 'train':
            ax.annotate('train', xy=(xmax - 40, ymin + 20))
            rect = patches.Rectangle((xmin, ymin), width, height, edgecolor='b', facecolor='none')
        elif row.objects == 'bicycle':
            ax.annotate('bicycle', xy=(xmax - 40, ymin + 20))
            rect = patches.Rectangle((xmin, ymin), width, height, edgecolor='g', facecolor='none')
        else:
            print("nothing")

        ax.add_patch(copy(rect))
        if not os.path.exists("imagesBox"):
            os.makedirs("imagesBox")

        fig.savefig('imagesBox/' + imageFileName, dpi=90, bbox_inches='tight')
    plt.close()
    print("ImageName: " + imageFileName + " is saved in imagesBox folder")

print("PLOTBOX COMPLETED!")