"""
Function: Assign clientID to train data
Input: Duplicated and Augmented train data from augment.py
Output: function returning client data given client ID

"""

import augment
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import collections
import matplotlib.pyplot as plt
import random
import simple_parser as sp
print('GPU', tf.test.is_gpu_available()) # TEST GPU AVAILABILITY

# create client id
NUM_CLIENTS = 100
# BUFFER_SIZE = len(dataset)
CID = list(range(0, NUM_CLIENTS ))

# for each client, randomly sample from the 16436 images
# keys = ['id', 'data']
client_data = []
client_num_pics = []
random.seed(603008)

# pick random subset to each client
for id in CID:
    num_pics = random.choice([5, 10, 100, 500])
    client_data.append(random.sample(augment.train_data, k = num_pics))
    client_num_pics.append(num_pics)

# see number of images per client
counter = collections.Counter(client_num_pics)
print(counter)
print("Total number of client images is ", np.sum(client_num_pics))

# define a function gives out client data receiving client id
def create_dataset_for_client(cid):
    assert cid in CID, "CANNOT FIND CLIENT DATA FOR SPECIFIED CID"
    return(client_data[cid])

# prelim analysis
# create a mapping from classes to count to each client
import seaborn as sns
classes = sp.dup_classes_count.keys()
for id in CID:
    ds = create_dataset_for_client(id)
    class_count = dict.fromkeys(classes, 0)
    for img in ds: # image in target client dataset
        for box in img['bboxes']: # all bbox in target client dataset
            c = box['class']
            if c in classes:
                class_count[c] = class_count[c] + 1
            else:
                print("CLASS NOT FOUND IN DICT")

    # plot histogram of class_count
    # print(class_count)
    names = list(class_count.keys())
    values = list(class_count.values())
    # print(values)
    # ax = sns.barplot(x=values, y=names)
    ax = plt.barh(names, values)
    # ax.set_title('Histogram of Bounding Box Classes For Client ID' + str(id))
    # ax.set_xlabel('Class Count')
    # ax.set_ylabel('Class Name')
    plt.savefig("clients/client-class-count-" + str(id) +".png" )
    print("SAVING FIGURE FOR CIIENT" + str(id))

###############????WHY PLOT GIVE DIFFERENT RESULTS????#############
##############POSSIBLE DUE TO PLOT WITHIN LOOP##################







