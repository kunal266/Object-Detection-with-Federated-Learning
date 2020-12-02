

def fedGen(data_gen_train):
    for each in data_gen_train:
        img = each[-2]
        x_img = cv2.imread(img['filepath'])
        y_img = img['bboxes']  # list of dict
        y_df = pd.DataFrame(y_img)  # df
        y_df.x1 = str(y_df.x1)  # int to str
        y_df.x2 = str(y_df.x2)
        y_df.y1 = str(y_df.y1)
        y_df.y2 = str(y_df.y2)

        x_tensor = tf.convert_to_tensor(x_img)
        y_tensor = tf.convert_to_tensor(y_df)
        id = tf.convert_to_tensor(each[-1]) # last element returned


        mydict = collections.OrderedDict()
        mydict['x_img'] = x_tensor
        mydict['y_img'] = y_tensor
        mydict['id'] = id
        yield mydict