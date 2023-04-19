#!/usr/bin/env python
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import dtypes
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference


def load_model(inputs, outputs):
    """ load .pb model
    """
    model_path = "./model/resnet50_v1.pb"
    graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.FastGFile(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    # optimized_graph_def = optimize_for_inference(graph_def, [item.split(':')[0] for item in inputs],
    #             [item.split(':')[0] for item in outputs], dtypes.float32.as_datatype_enum, False)
    g = tf.compat.v1.import_graph_def(graph_def, name='')

    return g


if __name__ == "__main__":
    input_shape=(224, 224, 3)

    ################
    # load image 
    orig_image = tf.keras.preprocessing.image.load_img(
        "./Weimaraner_wb.jpeg", target_size=input_shape)
    numpy_image = tf.keras.preprocessing.image.img_to_array(orig_image)
    numpy_image = np.expand_dims(numpy_image, axis=0).repeat(32, axis=0)  # batch images
    # numpy_image = np.expand_dims(numpy_image, axis=0)
    processed_image = tf.keras.applications.imagenet_utils.preprocess_input(
        numpy_image, mode="caffe"
    )
    ################

    sess_config = tf.ConfigProto()
    sess_config.inter_op_parallelism_threads = 0
    sess_config.intra_op_parallelism_threads = 0

    inputs = ["input_tensor:0"]
    outputs = ["ArgMax:0"]
    graph = load_model(inputs, outputs)

    with tf.Session(graph=graph, config=sess_config) as sess:
        pred = sess.run(outputs, feed_dict={inputs[0]: processed_image})
        print(pred)
    
    # best_time = 1000
    # for i in range(6):
    #     s = time.time()
    #     predictions = sess.run(outputs, feed_dict={inputs[0]: processed_image})
    #     e = time.time()
    #     if best_time > e - s:
    #         best_time = e - s
    #     print(f"inference: {e-s} s")

    # print(f"best inference time: {best_time:.4f}")
    # print("KPI: resnet " + str(10 / best_time))
