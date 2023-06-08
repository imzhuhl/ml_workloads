import tensorflow as tf
import random
import time
import argparse

import modeling


def generate_random_tensor(shape, vocab_size, rng=None, name=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return tf.constant(value=values, dtype=tf.int32, shape=shape, name=name)


def create_model_and_run():
    # bert-base default settings
    vocab_size = 30522
    type_vocab_size = 16

    # user defined
    batch_size = args.batch_size
    seq_length = args.seq_length

    config = modeling.BertConfig(vocab_size)

    # random input
    input_ids = generate_random_tensor([batch_size, seq_length], vocab_size)
    input_mask = generate_random_tensor([batch_size, seq_length], vocab_size=2)
    token_type_ids = generate_random_tensor([batch_size, seq_length], type_vocab_size)

    model = modeling.BertModel(
        config=config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=token_type_ids,
    )

    output = model.get_pooled_output()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        sess.run(output)

        time_list = []  # ms
        throughput_list = []
        for _ in range(args.steps):
            s = time.time()
            sess.run(output)
            e = time.time()
            time_list.append((e - s) * 1000)
            throughput_list.append(batch_size / (e - s))
        time_list.sort()
        throughput_list.sort()
        best_time = time_list[0]
        best_throughput = throughput_list[-1]
        tf.logging.info(f"[TARGET] [bert] [Time(ms)]: {best_time}")
        tf.logging.info(f"[TARGET] [bert] [Throughput(samples/sec)]: {best_throughput}")
        print(best_throughput)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        help='Batch size to train.',
                        type=int,
                        default=256)
    parser.add_argument('--seq_length',
                        help='length of sequence.',
                        type=int,
                        default=8)
    parser.add_argument('--steps',
                        help='Number of steps to run.',
                        type=int,
                        default=10)
    parser.add_argument('--inter',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--intra',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    create_model_and_run()
