import tensorflow as tf

def toy():
    for i in range(3):
        yield (i, i+1)

ds = tf.data.Dataset.from_generator(toy,
                                    (tf.int64, tf.int64),
                                    (tf.TensorShape(None), tf.TensorShape(None))
                                    )

def toy2():
    for i in range(3):
        yield {'a': i, 'b': i+1}

ds2 = tf.data.Dataset.from_generator(toy2,
                                    ({'a': tf.int64, 'b': tf.int64}),
                                     ({'a': tf.TensorShape(None), 'b': tf.TensorShape(None)})
                                    )

def toy3():
    for i in range(3):
        yield tf.data.Dataset.from_tensors({'a': i, 'b': i+1})

ds3 = tf.data.Dataset.from_generator(toy2,
                                    ({'a': tf.int64, 'b': tf.int64}),
                                     ({'a': tf.TensorShape(None), 'b': tf.TensorShape(None)})
                                    )

tf.ragged.constant({'a': 1, 'b': 2})