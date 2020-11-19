from codes.simple_parser import get_data
from codes import config, data_generators
from codes.augment import train_data
# from codes.vgg import get_img_output_length
from codes.simple_parser import val_imgs

# vgg not imported yet
# import vgg.get_img_output_length
def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)

# pass the settings from the command line, and persist them in the config object
C = config.Config()

train_imgs = train_data # duplicated and augmented train data

# create classes count of train imgs
classes_count = {}
for img in train_imgs:
    for box in img['bboxes']:
        class_name = box['class']
        if class_name not in classes_count:
            classes_count[class_name] = 1
        else:
            classes_count[class_name] += 1


random.shuffle(train_imgs)
num_imgs = len(train_imgs)

print(f'Num train samples {len(train_imgs)}')
print(f'Num val samples {len(val_imgs)}')

data_gen_train = data_generators.get_anchor_gt(train_imgs,  C, get_img_output_length,  mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, C, get_img_output_length, mode='val')

from codes.img_to_tensor import img_to_tensor
# generates ordered dict with tensor for each image
data_gen_train_tensor = img_to_tensor(data_gen_train[-1])


# try to assign 100 client ID to all train pictures
random.seed(603008)
imgs_cid = [] # list contain all cid for imgs
for i in range(len(train_imgs)):
    imgs_cid.append(random.randint(0, 100))
# Prelim analysis
# Number of img per client
from collections import Counter
print(Counter(imgs_cid))







