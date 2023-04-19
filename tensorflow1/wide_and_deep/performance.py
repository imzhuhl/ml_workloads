import tensorflow as tf
import time
import argparse
import os

# Definition of some constants
CONTINUOUS_COLUMNS = ['I' + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ['clicked']
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

    def _dnn(self, dnn_input, dnn_hidden_units=None, layer_name=''):
        for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
            with tf.variable_scope(layer_name + '_%d' % layer_id,
                                   reuse=tf.AUTO_REUSE) as dnn_layer_scope:
                dnn_input = tf.layers.dense(
                    dnn_input,
                    units=num_hidden_units,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.glorot_uniform_initializer(),
                    name=dnn_layer_scope)
        return dnn_input

    # create model
    def _create_model(self):
        # dnn part
        with tf.variable_scope('dnn'):
            # input layer
            with tf.variable_scope('input_from_feature_columns', reuse=tf.AUTO_REUSE):
                net = tf.feature_column.input_layer(
                    features=self._feature,
                    feature_columns=self._deep_column)
                
            # hidden layers
            with tf.variable_scope('dnn_layers', reuse=tf.AUTO_REUSE):
                net = self._dnn(net, self._dnn_hidden_units, 'hiddenlayer')
            
            # dnn logits
            with tf.variable_scope('logits') as dnn_logits_scope:
                dnn_logits = tf.layers.dense(
                    net, units=1, activation=None, name=dnn_logits_scope)

        # linear part
        with tf.variable_scope('linear') as scope:
            linear_logits = tf.feature_column.linear_model(
                units=1,
                features=self._feature,
                feature_columns=self._wide_column,
                sparse_combiner='sum',
                weight_collections=None,
                trainable=True)
        
        self._logits = tf.add_n([dnn_logits, linear_logits])
        self.probability = tf.math.sigmoid(self._logits)
        self.output = tf.round(self.probability)
    
    def _create_loss(self):
        self._logits = tf.squeeze(self._logits)
        self.loss = tf.losses.sigmoid_cross_entropy(
            self._label,
            self._logits,
            scope='loss',
            reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    
    def _create_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        dnn_optimizer = tf.train.AdamOptimizer(
                learning_rate=self._deep_learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
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
        
    def _create_metrics(self):
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self._label,
                                                    predictions=self.output)
        self.auc, self.auc_op = tf.metrics.auc(labels=self._label,
                                               predictions=self.probability,
                                               num_thresholds=1000)


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

            embedding_column = tf.feature_column.embedding_column(
                categorical_column,
                dimension=EMBEDDING_DIMENSIONS[column_name],
                combiner='mean')
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


def inference():
    # every input node's type
    type_enum_list = [tf.float32.as_datatype_enum for i in range(13)]
    type_enum_list.extend([tf.string.as_datatype_enum for i in range(26)])

    # read some real data
    test_file = os.path.join(args.data_location, "eval.csv")
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of test dataset is {}".format(no_of_test_examples))
    f_dict = {}
    with open(test_file) as f:
        for i in range(args.batch_size):
            line = f.readline()
            line = line.strip('\n')
            items = line.split(',')

            if i == 0:
                for name in LABEL_COLUMN + FEATURE_COLUMNS:
                    f_dict[f"{name}:0"] = []

            f_dict['clicked:0'].append(int(items[0]))
            for i in range(1, 14):
                f_dict[f"I{i}:0"].append(float(items[i]))

            for i in range(14, 40):
                f_dict[f"C{i-13}:0"].append(bytes(items[i], 'utf-8'))

    sess_config = tf.ConfigProto()
    sess_config.inter_op_parallelism_threads = args.inter
    sess_config.intra_op_parallelism_threads = args.intra

    # create feature column
    wide_column, deep_column = build_feature_columns()

    # define input placeholder
    inputs = {}
    for x in range(1, 14):
        inputs[f"I{x}"] = tf.placeholder(tf.float32, [None], name=f"I{x}")
    for x in range(1, 27):
        inputs[f"C{x}"] = tf.placeholder(tf.string, [None], name=f"C{x}")
    label = tf.placeholder(tf.int32, [None], name="clicked")
    real_input = [inputs, label]

    # create model
    model = WDL(wide_column=wide_column,
                deep_column=deep_column,
                inputs=real_input)

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([model.loss, model.train_op], feed_dict=f_dict)

        time_list = []  # ms
        throughput_list = []
        for _ in range(args.steps):
            s = time.time()
            sess.run([model.output], feed_dict=f_dict)
            e = time.time()
            time_list.append((e - s) * 1000)
            throughput_list.append(args.batch_size / (e - s))
        time_list.sort()
        throughput_list.sort()
        best_time = time_list[0]
        best_throughput = throughput_list[-1]
        print(f"[TARGET] [wide_and_deep] [Time(ms)]: {best_time}")
        print(f"[TARGET] [wide_and_deep] [Throughput(samples/sec)]: {best_throughput}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    parser.add_argument('--batch_size',
                        help='batch size',
                        type=int,
                        default=8000)
    parser.add_argument('--inter',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=8)
    parser.add_argument('--intra',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=8)
    parser.add_argument('--steps',
                        help='set the number of steps',
                        type=int,
                        default=10)
    args = parser.parse_args()

    inference()