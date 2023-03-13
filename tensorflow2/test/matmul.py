import argparse
import time
import tensorflow._api.v2.compat.v1 as tf
tf.disable_eager_execution()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inter", type=int, help="inter thread", default=0)
    parser.add_argument("--intra", type=int, help="intra thread", default=0)
    args = parser.parse_args()

    a = tf.random.uniform(shape=(4096, 4096), minval=-1, maxval=1, dtype=tf.float32)
    b = tf.random.uniform(shape=(4096, 4096), minval=-1, maxval=1, dtype=tf.float32)
    c = tf.matmul(a, b)

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=args.intra,
        inter_op_parallelism_threads=args.inter)

    with tf.Session(config=session_conf) as sess:
        for _ in range(10):
            s = time.time()
            y = sess.run(c)
            e = time.time()
            print(f"Time(ms): {(e-s)*1000}")
        print(y.shape)