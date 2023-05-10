#!/usr/bin/env python
import os
import time
import numpy as np
import threading
from queue import Queue
import tensorflow as tf
from tensorflow import dtypes
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference


def load_model(inputs, outputs):
    """ load .pb model
    """
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.inter_op_parallelism_threads = 0
    sess_config.intra_op_parallelism_threads = 0

    model_path = "./model/resnet50_v1.pb"
    graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.FastGFile(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    optimized_graph_def = optimize_for_inference(graph_def, [item.split(':')[0] for item in inputs],
                [item.split(':')[0] for item in outputs], dtypes.float32.as_datatype_enum, False)
    g = tf.compat.v1.import_graph_def(optimized_graph_def, name='')

    sess = tf.compat.v1.Session(graph=g, config=sess_config)
    return sess


def run_inference(sess, inputs, outputs, processed_image):
    for i in range(steps):
        pred = sess.run(outputs, feed_dict={inputs[0]: processed_image})
    return pred


# def handle_tasks(tasks_queue):
#     while True:
#         qitem = tasks_queue.get()
#         if qitem is None:
#             break
#         run_inference(qitem)
#         tasks_queue.task_done()


# class RunInference:
#     def __init__(self, sess, in_names, out_names, num_threads, processed_image):
#         self.sess = sess
#         self.in_names = in_names
#         self.out_names = out_names
#         self.num_threads = num_threads
#         self.processed_image = processed_image
#         self.tasks = Queue(maxsize=num_threads * 4)

#         self.workers = []
#         for i in range(num_threads):
#             t = threading.Thread(target=self.handle_tasks, args=(self.tasks,))
#             t.daemon = True
#             self.workers.append(t)
#             t.start()

#     def handle_tasks(self):
#         while True:
#             qitem = self.tasks.get()
#             if qitem is None:
#                 break
#             run_inference(qitem)
#             self.tasks.task_done()
    
#     def run(self):
#         pred = self.sess.run(self.out_names, feed_dict={self.in_names[0]: processed_image})
#         return pred



if __name__ == "__main__":
    input_shape=(224, 224, 3)
    batch_size = 32
    num_threads = 32
    steps = 10

    ################
    # load image 
    orig_image = tf.keras.preprocessing.image.load_img(
        "./data/Weimaraner_wb.jpeg", target_size=input_shape)
    numpy_image = tf.keras.preprocessing.image.img_to_array(orig_image)
    numpy_image = np.expand_dims(numpy_image, axis=0).repeat(batch_size, axis=0)  # batch images
    # numpy_image = np.expand_dims(numpy_image, axis=0)
    processed_image = tf.keras.applications.imagenet_utils.preprocess_input(
        numpy_image, mode="caffe"
    )
    ################

    
    inputs = ["input_tensor:0"]
    outputs = ["ArgMax:0"]
    sess = load_model(inputs, outputs)

    pred = sess.run(outputs, feed_dict={inputs[0]: processed_image})
    print(pred[0])


    t = []
    for i in range(num_threads):
        t.append(threading.Thread(target=run_inference, args=(sess, inputs, outputs, processed_image)))

    s = time.time()
    for i in range(num_threads):
        t[i].daemon = True
        t[i].start()
    
    for i in range(num_threads):
        t[i].join()
    e = time.time()
    print(f"Time: {e-s} s")
    print(f"Throughput: resnet " + str(num_threads * steps * batch_size / (e-s)))  # throughput

    # best_time = 1000
    # for i in range(6):
    #     s = time.time()
    #     predictions = sess.run(outputs, feed_dict={inputs[0]: processed_image})
    #     e = time.time()
    #     if best_time > e - s:
    #         best_time = e - s
    #     print(f"inference: {e-s} s")

        # pred = sess.run(["resnet_model/final_dense:0"], feed_dict={inputs[0]: processed_image})
        # print(pred[0][0, :5])
    
    sess.close()
    # print(f"best inference time: {best_time:.4f}")
    # print("TARGET: resnet " + str(batch_size / best_time))  # throughput
    
