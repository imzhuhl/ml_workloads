# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
import argparse
# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import os
import sys
import math
import collections
from tensorflow.python.client import timeline
import json
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python.ops import partitioned_variables

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

# Definition of some constants
CONTINUOUS_COLUMNS = ['I' + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ['clicked']
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
HASH_BUCKET_SIZES = {
    'C1': 2500,
    'C2': 2000,
    'C3': 300000,
    'C4': 250000,
    'C5': 1000,
    'C6': 100,
    'C7': 20000,
    'C8': 4000,
    'C9': 20,
    'C10': 100000,
    'C11': 10000,
    'C12': 250000,
    'C13': 40000,
    'C14': 100,
    'C15': 100,
    'C16': 200000,
    'C17': 50,
    'C18': 10000,
    'C19': 4000,
    'C20': 20,
    'C21': 250000,
    'C22': 100,
    'C23': 100,
    'C24': 250000,
    'C25': 400,
    'C26': 100000
}

EMBEDDING_DIMENSIONS = {
    'C1': 64,
    'C2': 64,
    'C3': 128,
    'C4': 128,
    'C5': 64,
    'C6': 64,
    'C7': 64,
    'C8': 64,
    'C9': 64,
    'C10': 128,
    'C11': 64,
    'C12': 128,
    'C13': 64,
    'C14': 64,
    'C15': 64,
    'C16': 128,
    'C17': 64,
    'C18': 64,
    'C19': 64,
    'C20': 64,
    'C21': 128,
    'C22': 64,
    'C23': 64,
    'C24': 128,
    'C25': 64,
    'C26': 128
}


class WDL():
    def __init__(self,
                 wide_column=None,
                 deep_column=None,
                 dnn_hidden_units=[1024, 512, 256],
                 optimizer_type='adam',
                 linear_learning_rate=0.2,
                 deep_learning_rate=0.01,
                 inputs=None,
                 bf16=False,
                 stock_tf=None,
                 adaptive_emb=False,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not inputs:
            raise ValueError("Dataset is not defined.")
        self._feature = inputs[0]
        self._label = inputs[1]

        self._wide_column = wide_column
        self._deep_column = deep_column
        if not wide_column or not deep_column:
            raise ValueError("Wide column or Deep column is not defined.")

        self.tf = stock_tf
        self.bf16 = False if self.tf else bf16
        self.is_training = True
        self._adaptive_emb = adaptive_emb

        self._dnn_hidden_units = dnn_hidden_units
        self._linear_learning_rate = linear_learning_rate
        self._deep_learning_rate = deep_learning_rate
        self._optimizer_type = optimizer_type
        self._input_layer_partitioner = input_layer_partitioner
        self._dense_layer_partitioner = dense_layer_partitioner

        self._create_model()
        with tf.name_scope('head'):
            self._create_loss()
            self._create_optimizer()
            self._create_metrics()

    # used to add summary in tensorboard
    def _add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                          tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)

    def _dnn(self, dnn_input, dnn_hidden_units=None, layer_name=''):
        for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
            with tf.variable_scope(layer_name + '_%d' % layer_id,
                                   partitioner=self._dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE) as dnn_layer_scope:
                dnn_input = tf.layers.dense(
                    dnn_input,
                    units=num_hidden_units,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.glorot_uniform_initializer(),
                    name=dnn_layer_scope)

                self._add_layer_summary(dnn_input, dnn_layer_scope.name)
        return dnn_input

    # create model
    def _create_model(self):
        # Dnn part
        with tf.variable_scope('dnn'):
            # input layer
            with tf.variable_scope('input_from_feature_columns',
                                   partitioner=self._input_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                if self._adaptive_emb and not self.tf:
                    '''Adaptive Embedding Feature Part 1 of 2'''
                    adaptive_mask_tensors = {}
                    for col in CATEGORICAL_COLUMNS:
                        adaptive_mask_tensors[col] = tf.ones([args.batch_size],
                                                             tf.int32)
                    net = tf.feature_column.input_layer(
                        features=self._feature,
                        feature_columns=self._deep_column,
                        adaptive_mask_tensors=adaptive_mask_tensors)
                else:
                    net = tf.feature_column.input_layer(
                        features=self._feature,
                        feature_columns=self._deep_column)
                self._add_layer_summary(net, 'input_from_feature_columns')

            # hidden layers
            dnn_scope = tf.variable_scope('dnn_layers', \
                partitioner=self._dense_layer_partitioner, reuse=tf.AUTO_REUSE)
            with dnn_scope.keep_weights(dtype=tf.float32) if self.bf16 \
                else dnn_scope:
                if self.bf16:
                    net = tf.cast(net, dtype=tf.bfloat16)

                net = self._dnn(net, self._dnn_hidden_units, 'hiddenlayer')

                if self.bf16:
                    net = tf.cast(net, dtype=tf.float32)

                # dnn logits
                logits_scope = tf.variable_scope('logits')
                with logits_scope.keep_weights(dtype=tf.float32) if self.bf16 \
                    else logits_scope as dnn_logits_scope:
                    dnn_logits = tf.layers.dense(net,
                                                 units=1,
                                                 activation=None,
                                                 name=dnn_logits_scope)
                    self._add_layer_summary(dnn_logits, dnn_logits_scope.name)

        # linear part
        with tf.variable_scope(
                'linear', partitioner=self._dense_layer_partitioner) as scope:
            linear_logits = tf.feature_column.linear_model(
                units=1,
                features=self._feature,
                feature_columns=self._wide_column,
                sparse_combiner='sum',
                weight_collections=None,
                trainable=True)

            self._add_layer_summary(linear_logits, scope.name)

        self._logits = tf.add_n([dnn_logits, linear_logits])
        self.probability = tf.math.sigmoid(self._logits)
        self.output = tf.round(self.probability)

    # compute loss
    def _create_loss(self):
        self._logits = tf.squeeze(self._logits)
        self.loss = tf.losses.sigmoid_cross_entropy(
            self._label,
            self._logits,
            scope='loss',
            reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        tf.summary.scalar('loss', self.loss)

    # define optimizer and generate train_op
    def _create_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        if self.tf or self._optimizer_type == 'adam':
            dnn_optimizer = tf.train.AdamOptimizer(
                learning_rate=self._deep_learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
        elif self._optimizer_type == 'adagrad':
            dnn_optimizer = tf.train.AdagradOptimizer(
                learning_rate=self._deep_learning_rate,
                initial_accumulator_value=0.1,
                use_locking=False)
        elif self._optimizer_type == 'adamasync':
            dnn_optimizer = tf.train.AdamAsyncOptimizer(
                learning_rate=self._deep_learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
        elif self._optimizer_type == 'adagraddecay':
            dnn_optimizer = tf.train.AdagradDecayOptimizer(
                learning_rate=self._deep_learning_rate,
                global_step=self.global_step)
        else:
            raise ValueError("Optimizer type error.")

        linear_optimizer = tf.train.FtrlOptimizer(
            learning_rate=self._linear_learning_rate,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0)
        train_ops = []
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_ops.append(
                dnn_optimizer.minimize(self.loss,
                                       var_list=tf.get_collection(
                                           tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope='dnn'),
                                       global_step=self.global_step))
            train_ops.append(
                linear_optimizer.minimize(self.loss,
                                          var_list=tf.get_collection(
                                              tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope='linear')))
            self.train_op = tf.group(*train_ops)

    # compute acc & auc
    def _create_metrics(self):
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self._label,
                                                    predictions=self.output)
        self.auc, self.auc_op = tf.metrics.auc(labels=self._label,
                                               predictions=self.probability,
                                               num_thresholds=1000)
        tf.summary.scalar('eval_acc', self.acc)
        tf.summary.scalar('eval_auc', self.auc)


# generate dataset pipline
def build_model_input(filename, batch_size, num_epochs):
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(filename))
        cont_defaults = [[0.0] for i in range(1, 14)]
        cate_defaults = [[' '] for i in range(1, 27)]
        label_defaults = [[0]]
        column_headers = TRAIN_DATA_COLUMNS
        record_defaults = label_defaults + cont_defaults + cate_defaults
        columns = tf.io.decode_csv(value, record_defaults=record_defaults)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = all_columns.pop(LABEL_COLUMN[0])
        features = all_columns
        return features, labels

    '''Work Queue Feature'''
    if args.workqueue and not args.tf:
        from tensorflow.python.ops.work_queue import WorkQueue
        work_queue = WorkQueue([filename])
        # For multiple files：
        # work_queue = WorkQueue([filename, filename1,filename2,filename3])
        files = work_queue.input_dataset()
    else:
        files = filename
    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(files)
    dataset = dataset.shuffle(buffer_size=20000,
                              seed=args.seed)  # fix seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=28)
    dataset = dataset.prefetch(2)
    return dataset


# generate feature columns
def build_feature_columns():
    # Notes: Statistics of Kaggle's Criteo Dataset has been calculated in advance to save time.
    mins_list = [
        0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    range_list = [
        1539.0, 22069.0, 65535.0, 561.0, 2655388.0, 233523.0, 26297.0, 5106.0,
        24376.0, 9.0, 181.0, 1807.0, 6879.0
    ]

    def make_minmaxscaler(min, range):
        def minmaxscaler(col):
            return (col - min) / range

        return minmaxscaler

    deep_columns = []
    wide_columns = []
    for column_name in FEATURE_COLUMNS:
        if column_name in CATEGORICAL_COLUMNS:
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                column_name, hash_bucket_size=10000, dtype=tf.string)
            wide_columns.append(categorical_column)

            if not args.tf:
                '''Feature Elimination of EmbeddingVariable Feature'''
                if args.ev_elimination == 'gstep':
                    # Feature elimination based on global steps
                    evict_opt = tf.GlobalStepEvict(steps_to_live=4000)
                elif args.ev_elimination == 'l2':
                    # Feature elimination based on l2 weight
                    evict_opt = tf.L2WeightEvict(l2_weight_threshold=1.0)
                else:
                    evict_opt = None
                '''Feature Filter of EmbeddingVariable Feature'''
                if args.ev_filter == 'cbf':
                    # CBF-based feature filter
                    filter_option = tf.CBFFilter(
                        filter_freq=3,
                        max_element_size=2**30,
                        false_positive_probability=0.01,
                        counter_type=tf.int64)
                elif args.ev_filter == 'counter':
                    # Counter-based feature filter
                    filter_option = tf.CounterFilter(filter_freq=3)
                else:
                    filter_option = None
                ev_opt = tf.EmbeddingVariableOption(
                    evict_option=evict_opt, filter_option=filter_option)

                if args.ev:
                    '''Embedding Variable Feature'''
                    categorical_column = tf.feature_column.categorical_column_with_embedding(
                        column_name, dtype=tf.string, ev_option=ev_opt)
                elif args.adaptive_emb:
                    '''                 Adaptive Embedding Feature Part 2 of 2
                    Expcet the follow code, a dict, 'adaptive_mask_tensors', is need as the input of 
                    'tf.feature_column.input_layer(adaptive_mask_tensors=adaptive_mask_tensors)'.
                    For column 'COL_NAME',the value of adaptive_mask_tensors['$COL_NAME'] is a int32
                    tensor with shape [batch_size].
                    '''
                    categorical_column = tf.feature_column.categorical_column_with_adaptive_embedding(
                        column_name,
                        hash_bucket_size=HASH_BUCKET_SIZES[column_name],
                        dtype=tf.string,
                        ev_option=ev_opt)
                elif args.dynamic_ev:
                    '''Dynamic-dimension Embedding Variable'''
                    print(
                        "Dynamic-dimension Embedding Variable isn't really enabled in model."
                    )
                    sys.exit()

            if args.tf or not args.emb_fusion:
                embedding_column = tf.feature_column.embedding_column(
                    categorical_column,
                    dimension=EMBEDDING_DIMENSIONS[column_name],
                    combiner='mean')
            else:
                '''Embedding Fusion Feature'''
                embedding_column = tf.feature_column.embedding_column(
                    categorical_column,
                    dimension=EMBEDDING_DIMENSIONS[column_name],
                    combiner='mean',
                    do_fusion=args.emb_fusion)

            deep_columns.append(embedding_column)
        else:
            normalizer_fn = None
            i = CONTINUOUS_COLUMNS.index(column_name)
            normalizer_fn = make_minmaxscaler(mins_list[i], range_list[i])
            column = tf.feature_column.numeric_column(
                column_name, normalizer_fn=normalizer_fn, shape=(1, ))
            wide_columns.append(column)
            deep_columns.append(column)

    return wide_columns, deep_columns


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return low_string == 'true'


# Get parse
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)
    parser.add_argument('--output_dir',
                        help='Full path to model output directory. \
                            Default to ./result. Covered by --checkpoint. '                                                                           ,
                        required=False,
                        default='./result')
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output. \
                            Default to ./result/$MODEL_TIMESTAMP'                                                                 ,
                        required=False)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--seed',
                        help='set the random seed for tensorflow',
                        type=int,
                        default=2021)
    parser.add_argument('--optimizer',
                        type=str, \
                        choices=['adam', 'adamasync', 'adagraddecay', 'adagrad'],
                        default='adam')
    parser.add_argument('--linear_learning_rate',
                        help='Learning rate for linear model',
                        type=float,
                        default=0.2)
    parser.add_argument('--deep_learning_rate',
                        help='Learning rate for deep model',
                        type=float,
                        default=0.01)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline. Default 0',
                        type=int,
                        default=0)
    parser.add_argument('--protocol',
                        type=str,
                        choices=['grpc', 'grpc++', 'star_server'],
                        default='grpc')
    parser.add_argument('--inter',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--intra',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--input_layer_partitioner', \
                        help='slice size of input layer partitioner, units MB. Default 8MB',
                        type=int,
                        default=8)
    parser.add_argument('--dense_layer_partitioner', \
                        help='slice size of dense layer partitioner, units KB. Default 16KB',
                        type=int,
                        default=16)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
                        action='store_true')
    parser.add_argument('--tf', \
                        help='Use TF 1.15.5 API and disable DeepRec feature to run a baseline.',
                        action='store_true')
    parser.add_argument('--smartstaged', \
                        help='Whether to enable smart staged feature of DeepRec, Default to True.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--emb_fusion', \
                        help='Whether to enable embedding fusion, Default to True.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--ev', \
                        help='Whether to enable DeepRec EmbeddingVariable. Default False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--ev_elimination', \
                        help='Feature Elimination of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'l2', 'gstep'],
                        default=None)
    parser.add_argument('--ev_filter', \
                        help='Feature Filter of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'counter', 'cbf'],
                        default=None)
    parser.add_argument('--op_fusion', \
                        help='Whether to enable Auto graph fusion feature. Default to True',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--micro_batch',
                        help='Set num for Auto Mirco Batch. Default close.',
                        type=int,
                        default=0)  #TODO: Defautl to True
    parser.add_argument('--adaptive_emb', \
                        help='Whether to enable Adaptive Embedding. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--dynamic_ev', \
                        help='Whether to enable Dynamic-dimension Embedding Variable. Default to False.',
                        type=boolean_string,
                        default=False)#TODO:enable
    parser.add_argument('--incremental_ckpt', \
                        help='Set time of save Incremental Checkpoint. Default 0 to close.',
                        type=int,
                        default=0)
    parser.add_argument('--workqueue', \
                        help='Whether to enable Work Queue. Default to False.',
                        type=boolean_string,
                        default=False)
    return parser


def load_data():
    # read some real data
    test_file = "./data.csv"
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of test dataset is {}".format(no_of_test_examples))
    data_dict = {}
    with open(test_file) as f:
        for i in range(args.batch_size):
            line = f.readline()
            line = line.strip('\n')
            items = line.split(',')

            if i == 0:
                for name in FEATURE_COLUMNS:
                    data_dict[name] =[]

            for i in range(1, 14):
                data_dict[f"I{i}"].append(float(items[i]))
            
            for i in range(14, 40):
                data_dict[f"C{i-13}"].append(bytes(items[i], 'utf-8'))

    return data_dict


def run(save=False):
    # define input placeholder
    inputs = {}
    for name in CONTINUOUS_COLUMNS:
        inputs[name] = tf.placeholder(tf.float32, [None], name=name)
    for name in CATEGORICAL_COLUMNS:
        inputs[name] = tf.placeholder(tf.string, [None], name=name)
    label = tf.placeholder(tf.int32, [None], name="clicked")
    real_input = [inputs, label]

    # create feature column
    wide_column, deep_column = build_feature_columns()

    # load data
    data_dict = load_data()

    f_dict = {}
    for k, v in data_dict.items():
        f_dict[f"{k}:0"] = v
    
    # create model
    model = WDL(wide_column=wide_column,
                deep_column=deep_column,
                linear_learning_rate=args.linear_learning_rate,
                deep_learning_rate=args.deep_learning_rate,
                optimizer_type=args.optimizer,
                bf16=args.bf16,
                stock_tf=args.tf,
                adaptive_emb=args.adaptive_emb,
                inputs=real_input)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for _ in range(10):
            s = time.time()
            pred = sess.run(['Sigmoid:0'], feed_dict=f_dict)
            e = time.time()
            print(f"Time(ms): {(e-s)*1000}")
        
        if save:
            tf.saved_model.simple_save(sess, "./saved_model", inputs=inputs, outputs={"probability": model.probability})

            frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph.as_graph_def(), ['Sigmoid'])
            tf.io.write_graph(frozen_graph, "./frozen_model", 'model.pb', as_text=False)


def run_from_saved_model():
    # load data
    data_dict = load_data()

    f_dict = {}
    for k, v in data_dict.items():
        f_dict[f"{k}:0"] = v
    
    run_metadata = tf.RunMetadata()
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    with tf.Session(graph=tf.Graph()) as sess:
        model = tf.saved_model.load(sess, ["serve"], "./saved_model")

        for _ in range(10):
            s = time.time()
            pred = sess.run(["Sigmoid:0"], feed_dict=f_dict)
            e = time.time()
            print(f"Time(ms): {(e-s)*1000}")

        pred = sess.run(["Sigmoid:0"], feed_dict=f_dict, options=options, run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_saved_model.json', 'w') as f:
            f.write(ctf)


def run_from_frozen_model():
    # load data
    data_dict = load_data()

    f_dict = {}
    for k, v in data_dict.items():
        f_dict[f"{k}:0"] = v
    
    # every input node's type
    type_enum_list = [tf.float32.as_datatype_enum for i in range(13)]
    type_enum_list.extend([tf.string.as_datatype_enum for i in range(26)])

    with tf.gfile.GFile("frozen_model/model.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph_def = optimize_for_inference(
            graph_def,
            FEATURE_COLUMNS,
            ["Sigmoid"],
            type_enum_list,
            False,
        )

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

        run_metadata = tf.RunMetadata()
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        with tf.Session(graph=graph) as sess:
            for _ in range(10):
                s = time.time()
                sess.run(['Sigmoid:0'], feed_dict=f_dict)
                e = time.time()
                print(f"Time(ms): {(e-s)*1000}")
            
            pred = sess.run(["Sigmoid:0"], feed_dict=f_dict, options=options, run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline_frozen_model.json', 'w') as f:
                f.write(ctf)


def freeze_model():
    # define input placeholder
    inputs = {}
    for x in range(1, 14):
        inputs[f"I{x}"] = tf.placeholder(tf.float32, [None], name=f"I{x}")
    for x in range(1, 27):
        inputs[f"C{x}"] = tf.placeholder(tf.string, [None], name=f"C{x}")

    label = tf.placeholder(tf.int32, [None], name="clicked")

    real_input = [inputs, label]

    # create feature column
    wide_column, deep_column = build_feature_columns()

    # create model
    model = WDL(wide_column=wide_column,
                deep_column=deep_column,
                linear_learning_rate=args.linear_learning_rate,
                deep_learning_rate=args.deep_learning_rate,
                optimizer_type=args.optimizer,
                bf16=args.bf16,
                stock_tf=args.tf,
                adaptive_emb=args.adaptive_emb,
                inputs=real_input)

    model.is_training = False
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # for op in sess.graph.get_operations():
        #     for t in op.values():
        #         print(t)
        # Get frozen graph
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), ['Sigmoid'])
        # Save the frozen graph
        tf.io.write_graph(frozen_graph, export_dir, 'model.pb', as_text=False)
        tf.io.write_graph(frozen_graph, export_dir, 'model.pbtxt', as_text=True)


def inference():
    feature_names = [
        "I1", "I2","I3","I4","I5","I6","I7","I8","I9","I10","I11","I12","I13",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13",
        "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"
    ]

    # every input node's type
    type_enum_list = [tf.float32.as_datatype_enum for i in range(13)]
    type_enum_list.extend([tf.string.as_datatype_enum for i in range(26)])

    # read some real data
    test_file = args.data_location + '/eval.csv'
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of test dataset is {}".format(no_of_test_examples))
    f_dict = {}
    with open(test_file) as f:
        for i in range(args.batch_size):
            line = f.readline()
            line = line.strip('\n')
            items = line.split(',')

            if i == 0:
                for name in feature_names:
                    f_dict[f"{name}:0"] =[]

            for i in range(1, 14):
                f_dict[f"I{i}:0"].append(float(items[i]))

            for i in range(14, 40):
                f_dict[f"C{i-13}:0"].append(bytes(items[i], 'utf-8'))

    run_metadata = tf.RunMetadata()
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    sess_config = tf.ConfigProto()
    sess_config.inter_op_parallelism_threads = args.inter
    sess_config.intra_op_parallelism_threads = args.intra

    with tf.Session(graph=tf.Graph(), config=sess_config) as sess:
        model = tf.saved_model.load(sess, ["serve"], os.path.join(export_dir, "1"))

        # warmup
        sess.run(["Sigmoid:0"], feed_dict=f_dict)

        time_list = []
        throughput_list = []
        for _ in range(10):
            s = time.time()
            # preds = sess.run(['Sigmoid:0'], feed_dict=f_dict)
            pred = sess.run(["Sigmoid:0"], feed_dict=f_dict, options=options, run_metadata=run_metadata)
            e = time.time()
            time_list.append((e - s) * 1000)
            throughput_list.append(args.batch_size / (e - s))
        time_list.sort()
        throughput_list.sort()
        avg_time = sum(time_list) / len(time_list)
        avg_throughput = sum(throughput_list) / len(throughput_list)
        print(f"[TARGET] [wide_and_deep] [Time(ms)]: {avg_time}")
        print(f"[TARGET] [wide_and_deep] [Throughput(samples/sec)]: {avg_throughput}")

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)


    # read freeze model and optimize for inference
    # with tf.gfile.GFile(os.path.join(export_dir, "./model.pb"), 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())

    # graph_def = optimize_for_inference(
    #     graph_def,
    #     feature_names,
    #     ["logits/Sigmoid"],
    #     type_enum_list,
    #     False,
    # )

    # with tf.Graph().as_default() as graph:
    #     tf.import_graph_def(graph_def, name="")
    #     with tf.Session(graph=graph, config=sess_config) as sess: 


def convert_graph_def_to_saved_model():
    # read freeze model and optimize for inference
    with tf.gfile.GFile(os.path.join("./forzen_model/model.pb"), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name="")
        inputs = {
            node.name: sess.graph.get_tensor_by_name(
                '{}:0'.format(node.name))
            for node in graph_def.node if node.op=='Placeholder'
        }
        outputs = {
            "Sigmoid": sess.graph.get_tensor_by_name('Sigmoid:0')
        }

        tf.saved_model.simple_save(sess, os.path.join("forzen_model", "1"), inputs=inputs, outputs=outputs)


def get_size(model_dir, model_file='saved_model.pb'):
    model_file_path = os.path.join(model_dir, model_file)
    print(model_file_path, '')
    pb_size = os.path.getsize(model_file_path)
    variables_size = 0
    if os.path.exists(
            os.path.join(model_dir,
                         'variables/variables.data-00000-of-00001')):
        variables_size = os.path.getsize(
            os.path.join(model_dir, 'variables/variables.data-00000-of-00001'))
        variables_size += os.path.getsize(
            os.path.join(model_dir, 'variables/variables.index'))
    print('Model size: {} KB'.format(round(pb_size / (1024.0), 3)))
    print('Variables size: {} KB'.format(round(variables_size / (1024.0), 3)))
    print('Total Size: {} KB'.format(
        round((pb_size + variables_size) / (1024.0), 3)))


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    run()
    # run(save=True)

    # run_from_saved_model()
    # run_from_frozen_model()
