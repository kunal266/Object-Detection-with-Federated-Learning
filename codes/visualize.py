"""
Function: visualize train image class histogram
Input: classes_count from simple_parser.get_data(annotate.txt)
Output: class-count.png

"""
import simple_parser as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#############BEFORE DUPLICATION################
names = list(sp.classes_count.keys())
values = list(sp.classes_count.values())

ax = sns.barplot(x = values, y = names)
ax.set_title('Histogram of Bounding Box Classes Before Duplication ')
ax.set_xlabel('Class Count')
ax.set_ylabel('Class Name')
plt.savefig("class-count.png")

###############AFTER DUPLICATION################
names = list(sp.dup_classes_count.keys())
values = list(sp.dup_classes_count.values())

ax = sns.barplot(x = values, y = names)
ax.set_title('Histogram of Bounding Box Classes After Duplication ')
ax.set_xlabel('Class Count')
ax.set_ylabel('Class Name')
plt.savefig("dup-class-count.png")