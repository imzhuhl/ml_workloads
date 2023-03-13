# import tensorflow.compat.v1 as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.client import timeline
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
import numpy as np
import time
import argparse
import os


def simple_net(X, Y):
    # Build the model
    dense1 = tf.layers.dense(X, 128, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dense1, 128, activation=tf.nn.relu)
    outputs = tf.layers.dense(dense2, 10, activation=tf.nn.softmax)

    # Define the loss function
    loss = tf.keras.losses.categorical_crossentropy(outputs, Y)

    # Define the optimizer
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    return optimizer, loss, outputs


def run(save=False):
    X = tf.placeholder(tf.float32, shape=(None, 32), name="Placeholder_X")
    Y = tf.placeholder(tf.float32, shape=(None, 10), name="Placeholder_Y")
    optimizer, loss, outputs = simple_net(X, Y)

    # Generate some dummy data
    np.random.seed(42)
    x_train = np.random.rand(args.batch_size, 32).astype(np.float32)
    y_train = np.random.rand(args.batch_size, 10).astype(np.float32)

    feed_dict = {
        "Placeholder_X:0": x_train,
        "Placeholder_Y:0": y_train,
    }

    sess_config = tf.ConfigProto()
    sess_config.inter_op_parallelism_threads = args.inter
    sess_config.intra_op_parallelism_threads = args.intra
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            s = time.time()
            _, _, pred = sess.run([optimizer, loss, outputs], feed_dict=feed_dict)
            e = time.time()
            print(f"Time(ms): {(e-s)*1000}")
        print(pred.shape)

        if save:
            # SavedModel format for tensorflow serving
            tf.saved_model.simple_save(sess, "./saved_model", inputs={"model_input": X}, outputs={"model_output": outputs})

            # Freeze model and save as a pb file
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph.as_graph_def(), ['dense_2/Softmax'])
            tf.io.write_graph(frozen_graph, './frozen_model', 'model.pb', as_text=False)


def run_from_saved_model():
    np.random.seed(42)
    x_train = np.random.rand(args.batch_size, 32).astype(np.float32)

    run_metadata = tf.RunMetadata()
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    with tf.Session(graph=tf.Graph()) as sess:
        model = tf.saved_model.load(sess, ["serve"], "./saved_model")
        
        for _ in range(10):
            s = time.time()
            sess.run(["dense_2/Softmax:0"], feed_dict={"Placeholder_X:0": x_train})
            e = time.time()
            print(f"Time(ms): {(e-s)*1000}")

        pred = sess.run(["dense_2/Softmax:0"], feed_dict={"Placeholder_X:0": x_train}, 
                        options=options, run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_saved_model.json', 'w') as f:
            f.write(ctf)


def run_from_frozen_model():
    np.random.seed(42)
    x_train = np.random.rand(args.batch_size, 32).astype(np.float32)

    with tf.gfile.GFile("frozen_model/model.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # graph_def = optimize_for_inference(
        #     graph_def,
        #     ["Placeholder_X"],
        #     ["dense_2/Softmax"],
        #     tf.float32.as_datatype_enum,
        #     False,
        # )

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

        run_metadata = tf.RunMetadata()
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        with tf.Session(graph=graph) as sess:
            for _ in range(10):
                s = time.time()
                sess.run(["dense_2/Softmax:0"], feed_dict={"Placeholder_X:0": x_train})
                e = time.time()
                print(f"Time(ms): {(e-s)*1000}")

            pred = sess.run(["dense_2/Softmax:0"], feed_dict={"Placeholder_X:0": x_train}, 
                            options=options, run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline_frozen_model.json', 'w') as f:
                f.write(ctf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch size", default=512)
    parser.add_argument("--inter", type=int, default=0)
    parser.add_argument("--intra", type=int, default=0)
    args = parser.parse_args()

    run()

    # run_from_saved_model()
    # run_from_frozen_model()
