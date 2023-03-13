import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_eager_execution()

import numpy as np
import time
import argparse


def conv_net(X):
    # Build the model
    conv1 = tf.layers.conv2d(X, 64, 3, padding='same', activation=tf.nn.relu)
    outputs = tf.layers.conv2d(conv1, 128, 3, padding='same', activation=tf.nn.relu)
    return outputs


def run():

    X = tf.placeholder(tf.float32, shape=(None, 100, 100, 3), name="Placeholder_X")
    outputs = conv_net(X)

    # Generate some dummy data
    np.random.seed(42)
    x_train = np.random.rand(args.batch_size, 100, 100, 3).astype(np.float32)

    feed_dict = {
        "Placeholder_X:0": x_train,
    }
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            s = time.time()
            pred = sess.run(outputs, feed_dict=feed_dict)
            e = time.time()
            print(f"Time(ms): {(e-s)*1000}")
        print(pred.shape)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch size", default=512)
    args = parser.parse_args()
    run()