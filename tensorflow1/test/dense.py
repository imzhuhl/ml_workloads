import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
import numpy as np
import time
import argparse
import os


def run():
    X = tf.placeholder(tf.float32, shape=(None, 1024), name="Placeholder_X")
    # dense layer
    dense = tf.layers.dense(X, 1024, 
                            kernel_initializer=tf.glorot_uniform_initializer(),
                            bias_initializer=tf.glorot_uniform_initializer())
    

    # Generate some dummy data
    np.random.seed(42)
    x_train = np.random.rand(args.batch_size, 1024).astype(np.float32)

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=args.intra,
        inter_op_parallelism_threads=args.inter)

    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            s = time.time()
            y = sess.run(dense, feed_dict={"Placeholder_X:0": x_train})
            e = time.time()
            print(f"Time(ms): {(e-s)*1000}")
        print(y.shape)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch size", default=4000)
    parser.add_argument("--inter", type=int, default=0)
    parser.add_argument("--intra", type=int, default=0)
    args = parser.parse_args()

    run()
