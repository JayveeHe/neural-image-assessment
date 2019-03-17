# coding=utf-8

"""
Created by jayveehe on 2019/2/13.
http://code.dianpingoa.com/hejiawei03
"""
import os
import random
import sys

import cPickle

import StringIO

import datetime, time
import requests
import tensorflow as tf
import urllib3
from PIL import Image
import keras
from keras import backend as K, Input, optimizers
from keras.applications.imagenet_utils import preprocess_input
from keras.backend import binary_crossentropy
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.contrib import slim
from tensorflow.python.framework.errors_impl import InvalidArgumentError


from tensorflow.contrib.slim.nets import inception

import nima

slim = tf.contrib.slim

urllib3.disable_warnings()

PARENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PARENT_PATH)
PROJECT_PATH = PARENT_PATH
print('Relate file: %s, \tPROJECT path = %s\nPARENT PATH = %s' % (__file__, PROJECT_PATH, PARENT_PATH))

# from fine_tune_tf import load_img_as_array_from_Gfile
# from inception_resnet_v2 import InceptionResNetV2

# parser = argparse.ArgumentParser()
flags = tf.app.flags
FLAGS = flags.FLAGS
K.set_image_dim_ordering('tf')

# parser = argparse.ArgumentParser()
flags = tf.app.flags
FLAGS = flags.FLAGS
K.set_image_dim_ordering('tf')

flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.logging.set_verbosity(tf.logging.INFO)
flags.DEFINE_integer("batch_size", 20, "Number of samples in a batch.")
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
# parser.add_argument('--train_data', type=str,
#                     default='hdfs://localhost:50070/tf_data/minist/train.tfrecords',
#                     help='Directory for storing input data')
flags.DEFINE_string("output_dir", "output_models", "One of 'ps', 'worker'")
flags.DEFINE_string("log_dir", "output_models", "One of 'ps', 'worker'")

flags.DEFINE_integer("max_epoch", 20, "Number of samples in a batch.")
flags.DEFINE_integer("iter_per_epoch", 10, "Number of samples in a batch.")
flags.DEFINE_integer("valid_iter", 10, "Number of samples in a valid batch.")

flags.DEFINE_boolean('is_hdfs', False, 'is hdfs model')
flags.DEFINE_integer("worker_nums", 1, "Number of samples in a batch.")
# flags.DEFINE_integer("workers", 1, "Number of samples in a batch.")
flags.DEFINE_string("weights_path", "inception_resnet", "One of 'ps', 'worker'")
flags.DEFINE_string("pkl_weights_path", "inception_resnet", "One of 'ps', 'worker'")
flags.DEFINE_string("weights_meta_path", "inception_resnet", "One of 'ps', 'worker'")
flags.DEFINE_string("weights_ckpt_path", "inception_resnet", "One of 'ps', 'worker'")
flags.DEFINE_string("pic_size", "299_299", "pic size like 299_299")
flags.DEFINE_string("train_tag", "Default", "list of undefined args")
flags.DEFINE_string('w2vdict_path', '', 'path of word2vec dict file')

flags.DEFINE_string("undefork", "", "list of undefined args")
FLAGS(sys.argv)

output_dir = FLAGS.output_dir
if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
max_epoch = FLAGS.max_epoch
weights_path = FLAGS.weights_path
learning_rate = FLAGS.learning_rate
is_hdfs = FLAGS.is_hdfs
worker_nums = FLAGS.worker_nums
batch_size = FLAGS.batch_size
print 'batch_size=%s' % batch_size
print  FLAGS.job_name
tf.logging.info('is_hdfs: %s' % is_hdfs)
tf.logging.info('learning rate: %s' % FLAGS.learning_rate)
# is_hdfs = True
train_tag = FLAGS.train_tag
tf.logging.info('training tag: %s' % train_tag)
# random.seed(20180723)
rand = random.Random(20180723)
if is_hdfs or FLAGS.job_name == 'ps':
    # train_img_path_list = open('train_img_path.txt', 'r').read().split('\n')
    # test_img_path_list = open('test_img_path.txt', 'r').read().split('\n')
    # tf.gfile.ListDirectory()
    # train_data_root_path = 'viewfs://hadoop-meituan/user/hadoop-dpsr/hejiawei03/tfrecords/pairwise_tfrecords_newtaggedV11_trainset_newsample/'
    train_data_root_path = 'viewfs://hadoop-meituan/ghnn01/user/hadoop-dpsr/zhouxin25/picprocess/tfrecords/ava_distribute_traindata'
    tfrecord_name_list = tf.gfile.ListDirectory(train_data_root_path)
    rand.shuffle(tfrecord_name_list)
    # train_tfrecord_path_list = [train_data_root_path + a for a in
    #                             train_tfrecord_name_list]
    total_tfrecord_num = len(tfrecord_name_list)
    train_num = int(total_tfrecord_num * 0.8)
    tfrecord_path_list = [os.path.join(train_data_root_path, a) for a in
                          tfrecord_name_list]
    train_tfrecord_path_list = tfrecord_path_list[:train_num]
    test_tfrecord_path_list = tfrecord_path_list[train_num:]
    # test_data_root_path = 'viewfs://hadoop-meituan/user/hadoop-dpsr/hejiawei03/tfrecords/pairwise_tfrecords_newtaggedV11_validset_newsample/'
else:
    train_root = '/data/hejiawei03/img/train'
    train_plist = os.listdir(train_root)
    train_tfrecord_path_list = [os.path.join(train_root, a) for a in train_plist]
    test_root = '/data/hejiawei03/img/test'
    test_plist = os.listdir(test_root)
    test_tfrecord_path_list = [os.path.join(test_root, a) for a in test_plist]
    train_tf_records_path_list = []
    test_tf_records_path_list = []
    # train_tf_records_path_list = tf.gfile.GFile(
    #     'viewfs://hadoop-meituan/ghnn05/hadoop-dpsr/hejiawei03/tf_records_pathlist.txt').read().split('\n')[:-10]
    # test_tf_records_path_list = tf.gfile.GFile(
    #     'viewfs://hadoop-meituan/ghnn05/hadoop-dpsr/hejiawei03/tf_records_pathlist.txt').read().split('\n')[-10:]
train_block_size = len(train_tfrecord_path_list) / worker_nums
test_block_size = len(test_tfrecord_path_list) / worker_nums
train_tfrecord_block_size = int(np.ceil(len(train_tfrecord_path_list) / worker_nums))
test_tfrecord_block_size = int(np.ceil(len(test_tfrecord_path_list) / worker_nums))
tf.logging.info(FLAGS)
VALID_ITERS = FLAGS.valid_iter
WIDTH, HEIGHT = FLAGS.pic_size.strip().split('_')
WIDTH = int(WIDTH)
HEIGHT = int(HEIGHT)

begin_train_index = int(FLAGS.task_index * train_tfrecord_block_size)
end_train_index = int((FLAGS.task_index + 1) * train_tfrecord_block_size)
cur_train_tfrecord_list = train_tfrecord_path_list[begin_train_index:end_train_index]
# cur_train_tfrecord_list = train_tfrecord_path_list[begin_train_index:2]

begin_test_index = int(FLAGS.task_index * test_tfrecord_block_size)
# end_test_index = int((FLAGS.task_index + 1) * test_tfrecord_block_size)
# end_test_index = int(len(test_tfrecord_path_list) * 0.2)
end_test_index = int(len(test_tfrecord_path_list))
cur_test_tfrecord_list = test_tfrecord_path_list[begin_test_index:end_test_index]
# cur_test_tfrecord_list = train_tfrecord_path_list[begin_train_index:2]

tf.logging.info('train tfrecord index: %s -> %s' % (begin_train_index, end_train_index))
tf.logging.info('train tfrecord list: %s' % cur_train_tfrecord_list)

tf.logging.info('test tfrecord index: %s -> %s' % (begin_test_index, end_test_index))
tf.logging.info('test tfrecord list: %s' % cur_test_tfrecord_list)


def sigmoid_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)


# pairwise model initiation
def load_model_weights(pkl_path, model):
    print 'loading model from %s' % pkl_path
    wfin = tf.gfile.GFile(pkl_path, 'rb')
    wlist = cPickle.load(wfin)
    model.set_weights(wlist)
    wfin.close()
    return model


def _get_init_fn():
    """Return a function that 'warm-starts' the training.

    Returns:
      An init function.
    """
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from {}'.format(checkpoint_path))

    return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)


def get_model(images, num_classes=10, is_training=True, weight_decay=4e-5,
              dropout_keep_prob=0.75):
    """Neural Image Assessment from https://arxiv.org/abs/1709.05424

    Talebi, Hossein, and Peyman Milanfar. "NIMA: Neural Image Assessment."
    arXiv preprint arXiv:1709.05424 (2017).

    Args:
      images: a tensor of shape [batch_size, height, width, channels].
      num_classes: number of predicted classes. Defaults to 10.
      is_training: whether is training or not.
      weight_decay: the weight decay to use for regularizing the model.
      dropout_keep_prob: the percentage of activation values that are retained.
        Defaults to 0.75

    Returns:
      predictions: a tensor of size [batch_size, num_classes].
      end_points: a dictionary from components of the network.
    """
    arg_scope = inception.inception_v2_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        logits, end_points = inception.inception_v2(
            images, num_classes, is_training=is_training,
            dropout_keep_prob=dropout_keep_prob)

    predictions = tf.nn.softmax(logits)

    return predictions, end_points


def input_fn(tfrecord_path_list, split_name, batch_size, workers):
    """Load and preprocess a dataset.

    Args:
      dataset_dir: path to the TFRecord files.
      split_name: train or validation split.
      batch_size: number of items per batch.
      workers: the number of parallel preprocessing workers to run.

    Returns:
      A tuple of Dataset iterator and a number of examples.
    """

    def preprocess(example, num_classes=10, is_training=True):
        """Extract and preprocess dataset features.
    
        Args:
          example: an instance of protobuf-encoded example.
          num_classes: number of predicted classes. Defaults to 10.
          is_training: whether is training or not.
    
        Returns:
          A tuple of `image` and `scores` tensors.
        """
        features = {'scores': tf.VarLenFeature(tf.float32),
                    'image': tf.FixedLenFeature((), tf.string)}
        parsed = tf.parse_single_example(example, features)
        image = tf.image.decode_jpeg(parsed['image'], channels=3)
        image = nima.preprocess_image(image, is_training=is_training)
        scores = parsed['scores']
        scores = tf.sparse_tensor_to_dense(scores)
        scores = tf.reshape(scores, [num_classes])
        scores = scores / tf.reduce_sum(scores, axis=-1, keepdims=True)
        return image, scores

    # folder = os.path.join(dataset_dir, '{}_*.tfrecord'.format(split_name))
    # filenames = tf.data.Dataset.list_files(folder)
    dataset = tf.data.TFRecordDataset(tfrecord_path_list)
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()
    dataset = dataset.map(preprocess, num_parallel_calls=workers)
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(2)

    # filename = '{}.txt'.format(split_name)
    # with open(os.path.join(dataset_dir, filename), 'r') as f:
    #     examples = int(f.read().strip())
    return dataset
    # data_iterator = dataset.make_one_shot_iterator()
    # batch_features, batch_labels = data_iterator.get_next()
    # return batch_features, batch_labels


class Trainer(object):
    def __init__(self):
        # 定义集群相关
        if len(FLAGS.ps_hosts) > 0:
            self.ps_hosts = FLAGS.ps_hosts.split(',')
            tf.logging.info('PS hosts are: %s' % self.ps_hosts)
            self.worker_hosts = FLAGS.worker_hosts.split(',')
            tf.logging.info('Worker hosts are: %s' % self.worker_hosts)
            self.cluster = tf.train.ClusterSpec({"ps": self.ps_hosts, "worker": self.worker_hosts})
            self.num_ps = self.cluster.num_tasks('ps')
        else:
            self.worker_hosts = FLAGS.worker_hosts.split(',')
            tf.logging.info('Worker hosts are: %s' % self.worker_hosts)
            self.cluster = tf.train.ClusterSpec({"worker": self.worker_hosts})
            self.num_ps = 0
        tf.logging.info('cluster init!')
        self.ps_hosts = FLAGS.ps_hosts.split(",")
        self.worker_hosts = FLAGS.worker_hosts.split(",")
        tf.logging.info('PS hosts are: %s' % self.ps_hosts)
        tf.logging.info('Worker hosts are: %s' % self.worker_hosts)
        self.job_name = FLAGS.job_name
        self.task_index = FLAGS.task_index
        # self.cluster = tf.train.ClusterSpec({"ps": self.ps_hosts, "worker": self.worker_hosts})
        self.server = tf.train.Server(self.cluster, job_name=self.job_name, task_index=self.task_index)
        tf.logging.info('server init!')

        # 定义是否是chief worker（首席工人）
        self.is_chief = (self.task_index == 0 and self.job_name == 'worker')
        worker_prefix = '/job:worker/task:%s' % self.task_index
        self.cpu_device = '%s/cpu:0' % worker_prefix
        self.gpu_device = '%s/gpu:0' % worker_prefix
        self.param_server_device = tf.train.replica_device_setter(
            worker_device=self.gpu_device, cluster=self.cluster)
        # self.num_ps = self.cluster.num_tasks('ps')
        self.num_worker = self.cluster.num_tasks('worker')
        self.sv = None
        self.cur_epoch = 0
        tf.logging.info('Trainer init!')

    def create_sv(self, saver=None):
        sv = tf.train.Supervisor(
            is_chief=self.is_chief,
            logdir=FLAGS.log_dir,
            saver=saver if saver else tf.train.Saver(),
            global_step=self.global_step)
        self.sv = sv

    def create_session(self):
        # graph_options = tf.GraphOptions(enable_bfloat16_sendrecv=False)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=False,
                                     # graph_options=graph_options
                                     )
        sess_config.gpu_options.allow_growth = True
        tf.logging.info('server target=====%s' % self.server.target)
        return tf.Session(target=self.server.target, config=sess_config)
        # return MonitoredTrainingSession(master=self.server.target, config=sess_config, is_chief=self.is_chief,
        #                                 checkpoint_dir=FLAGS.output_dir, hooks=hooks)

    def train(self):
        if self.job_name == "ps":
            with tf.device('/cpu:0'):
                self.server.join()
        elif self.job_name == "worker":
            with tf.device(self.param_server_device):
                # with tf.device(self.gpu_device):
                tf.logging.info('using device: %s' % self.param_server_device)
                self.global_step = tf.train.get_or_create_global_step()
                # define you variables
                tf.logging.info('data generator init!')

                with self.create_session() as sess:
                    K.set_session(sess)
                    K.set_learning_phase(1)
                    # sess.graph._unsafe_unfinalize()

                    tf.logging.info('Init model weights using file: %s' % weights_path)
                    with tf.name_scope("querymatch_model"):
                        train_dataset = input_fn(cur_train_tfrecord_list, 'train', FLAGS.batch_size, workers=5)
                        # train_dataset = read_from_tfrecord(
                        #     cur_train_tfrecord_list,
                        #     batchsize=FLAGS.batch_size, w2v_manager=w2v_m)
                        common_dataset_iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                                              train_dataset.output_shapes)
                        img_objs, scores = common_dataset_iter.get_next()
                        # labels = label_batch
                        train_initializer = common_dataset_iter.make_initializer(train_dataset)
                        # pic1_input = Input(batch_shape=(None, 224, 224, 3), name='pic1_raw_input')
                        pic_input = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name='pic_raw_input')
                        # wordvec_input = tf.placeholder(dtype=tf.float32, shape=(None, 300), name='wordvec_input')
                        labels = tf.placeholder(tf.float32, [None, 1])
                        predictions, end_points = get_model(pic_input, is_training=True, )
                        # input_wordvec, input_pic, fc300, sim_score, trainable_vars_list, resnet_model = build_network(
                        #     pkl_path=FLAGS.pkl_weights_path)
                        # trainable_vars_list = tf.trainable_variables('nima/trainable')
                        # tf.logging.info('trainable vars list : %s' % trainable_vars_list)
                        # collection settings
                        # tf.add_to_collection('pairwise_predictions', raw_pic1_batch)
                        # tf.add_to_collection('pairwise_predictions', raw_pic2_batch)
                        trainable_vars_list = slim.get_trainable_variables()
                        tf.add_to_collection('nima_predictions', pic_input)
                        tf.add_to_collection('nima_predictions', predictions)
                        tf.add_to_collection('nima_predictions', labels)
                        for tvar in trainable_vars_list:
                            tf.add_to_collection('querymatch_trainable_vars', tvar)
                    valid_dataset = input_fn(cur_test_tfrecord_list, 'valid', FLAGS.batch_size, workers=5)
                    valid_initializer = common_dataset_iter.make_initializer(valid_dataset)
                    loss = nima.emd_loss(scores, predictions, r=2)
                    tf.losses.add_loss(loss)

                    learning_rate_obj = tf.train.exponential_decay(
                        FLAGS.learning_rate, self.global_step, 5,
                        0.01, staircase=True,
                        name='exponential_decay_learning_rate')

                    # optimizer = tf.train.AdamOptimizer(learning_rate_obj)
                    total_loss = tf.losses.get_total_loss()
                    # loss = cosine_loss
                    # loss = tf.reduce_sum(binary_crossentropy(labels, predictions))
                    # opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                    #                                 momentum=0.9, decay=0.0, epsilon=1e-06)  # tf opt
                    # opt = optimizers.RMSprop(learning_rate, rho=0.9, epsilon=1e-06)  # keras opt
                    opt = tf.train.AdamOptimizer(learning_rate=learning_rate_obj)
                    train_op = opt.minimize(total_loss,
                                            var_list=trainable_vars_list,
                                            global_step=self.global_step)
                    # calc_acc = sigmoid_acc(labels, sim_score)
                    # calc_auc, auc_op = tf.metrics.auc(labels, sim_score)
                    correlation = slim.metrics.streaming_pearson_correlation(
                        predictions, scores)
                    tf.logging.info('start setting weights using file: %s' % FLAGS.pkl_weights_path)
                    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                    sess.run(init)
                    # r = sess.run(resoutput, feed_dict={input_pic: preprocess_input(np.array([arr]))})
                    # tf.summary.scalar('iter_auc', tf.reduce_mean(calc_auc))
                    tf.summary.scalar('iter_loss', loss)
                    for idx in xrange(len(trainable_vars_list)):
                        tf.logging.info('adding hist: %s' % trainable_vars_list[idx].name)
                        tf.summary.histogram('trainable_histogram_%s' % trainable_vars_list[idx].name,
                                             trainable_vars_list[idx])

                    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
                    merged = tf.summary.merge_all()
                    epoch_loss = tf.placeholder(dtype=tf.float32, shape=(None,), name='epoch_loss')
                    epoch_loss_summary = tf.summary.scalar('epoch_loss', tf.reduce_mean(epoch_loss))
                    epoch_correlation = tf.placeholder(dtype=tf.float32, shape=(None,), name='epoch_correlation')
                    epoch_correlation_summary = tf.summary.scalar('epoch_correlation', tf.reduce_mean(epoch_correlation))
                    epoch_merge = tf.summary.merge(
                        [epoch_loss_summary])
                    # image1_scalar = tf.summary.image('image', input_pic[:1], max_outputs=5)
                    # image_label_scalar = tf.summary.scalar('match_score', tf.reduce_mean(labels[:1]))
                    # image_pred_scalar = tf.summary.scalar('match_pred', tf.reduce_mean(sim_score[:1]))
                    # image_merge = tf.summary.merge(
                    #     [image1_scalar, image_label_scalar, image_pred_scalar])
                    if self.is_chief:
                        str_date = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
                        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/%s_train_%s' % (train_tag, str_date),
                                                             sess.graph)
                        valid_writer = tf.summary.FileWriter(FLAGS.log_dir + '/%s_valid_%s' % (train_tag, str_date),
                                                             sess.graph)
                        # epoch_writer = tf.summary.FileWriter(FLAGS.log_dir + '/epoch_%s' % str_date,
                        #                                      sess.graph)
                        # img_writer = tf.summary.FileWriter(FLAGS.log_dir + '/%s_img_%s' % (train_tag, str_date),
                        #                                    sess.graph)
                    tf.logging.info('start training')
                    # for early stopping
                    min_epoch_valid_loss = 10000
                    max_epoch_valid_auc = 0
                    best_epoch = 0
                    patience_count = 0
                    TOTAL_PATIENCE = 10
                    # try:
                    self.saver = tf.train.Saver()
                    is_warmup = False
                    is_sync_train = False
                    while self.cur_epoch < FLAGS.max_epoch:
                        try:
                            iter_index = 0
                            train_epoch_loss = []
                            train_epoch_correlation = []
                            train_epoch_acc = []
                            train_epoch_labels = []
                            train_epoch_preds = []
                            sess.run([train_initializer])
                            tf.logging.info('new epoch inited')
                            K.set_learning_phase(1)
                            # --------- warm up ------------
                            # train_op, is_warmup = self.get_warmup_opt(loss, self.cur_epoch, trainable_vars_list,
                            #                                           train_op, sess)

                            while True:
                                waiting_start_time = time.time()
                                waiting_end_time = time.time()
                                read_data_start = time.time()
                                try:
                                    pic_batch, cur_label_batch = sess.run(
                                        [img_objs, scores])
                                except InvalidArgumentError, iae:
                                    tf.logging.error('[training]failed to fetch batch data')
                                    continue
                                read_data_done = time.time()
                                train_start_time = time.time()
                                # acc_list = []
                                # iter_loss = 0
                                # preds = []
                                # gs = iter_index

                                _, iter_loss, iter_consine_loss, gs, preds, summary = sess.run(
                                    [train_op, loss, self.global_step, merged],
                                    feed_dict={pic_input: pic_batch,
                                               labels: cur_label_batch})
                                # _ = sess.run(resoutput, feed_dict={pic1_input: pic1_batch})

                                train_end_time = time.time()
                                # fill in AUC data
                                train_epoch_labels.extend(cur_label_batch)
                                train_epoch_preds.extend(preds)
                                tmp_pred_str = 'pred 1 count: %s, total: %s' % (
                                    len(filter(lambda x: x > 0.5, preds)), len(preds))
                                tf.logging.info(
                                    '[%s] epoch %s iter %s global step: %s - | %s batch loss = %s.'
                                    ' Throughput %s samples/sec, data loading cost: %s, train cost: %s, wait cost: %s' % (
                                        datetime.datetime.now(), self.cur_epoch, iter_index, gs, tmp_pred_str,
                                        iter_loss,
                                        batch_size / (train_end_time - read_data_start),
                                        (read_data_done - read_data_start), (train_end_time - train_start_time),
                                        (waiting_end_time - waiting_start_time)))
                                # _, train_loss = sess.run([train_op, loss])
                                if iter_index % 20 == 0 and self.is_chief:
                                    train_writer.add_summary(summary, gs)
                                    # img_writer.add_summary(img_summary, gs)
                                waiting_start_time = time.time()
                                train_epoch_loss.append(iter_loss)
                                iter_index += 1
                        except tf.errors.OutOfRangeError:
                            tf.logging.info('training -- iter limit reached')
                            self.cur_epoch += 1
                            # validation
                            tf.logging.info(
                                'epoch done! epoch loss=%s' % (
                                    np.mean(train_epoch_loss)))
                            if self.is_chief:
                                K.set_learning_phase(0)
                                mse_list = []
                                pred_list = []
                                label_list = []
                                valid_loss_list = []
                                last_epoch = self.cur_epoch - 1
                                tf.logging.info('train epoch label size:%s' % len(train_epoch_labels))
                                # keras_epoch_auc = auc(train_epoch_labels, train_epoch_preds)
                                calc_epoch_correlation = sess.run([correlation],
                                                                           feed_dict={
                                                                               labels: np.array(train_epoch_labels),
                                                                               predictions: np.array(
                                                                                   train_epoch_preds)})
                                tf.logging.info('Epoch %s correlation: %s' % (
                                    self.cur_epoch, calc_epoch_correlation))
                                train_epoch_summary = sess.run(epoch_merge,
                                                               feed_dict={epoch_loss: np.array(train_epoch_loss),
                                                                          })
                                train_writer.add_summary(train_epoch_summary, last_epoch)
                                try:
                                    sess.run([valid_initializer])
                                    tf.logging.info('start validation')
                                    epoch_valid_labels = []
                                    epoch_valid_preds = []
                                    for valid_iter_index in range(VALID_ITERS):
                                        # tmp_valid_pic1_batch, tmp_valid_pic2_batch, tmp_valid_label_batch = sess.run(
                                        #     [valid_pic1_batch, valid_pic2_batch, valid_label_batch])
                                        try:
                                             valid_pic_batch, valid_label_batch = sess.run(
                                                [img_objs, scores])
                                        except InvalidArgumentError, iae:
                                            tf.logging.error('[validation]failed to fetch batch data')
                                            continue
                                        preds, val_loss, val_correlation, gs, valid_summary = sess.run(
                                            [predictions, loss, calc_epoch_correlation, self.global_step, merged],
                                            feed_dict={pic_input: valid_pic_batch,
                                                       labels: valid_label_batch}
                                        )
                                        # fill in valid AUC data
                                        epoch_valid_labels.extend(valid_label_batch)
                                        epoch_valid_preds.extend(preds)
                                        pred_list.extend(preds)
                                        valid_loss_list.append(val_loss)
                                        label_list.extend(valid_label_batch)
                                        # tf.logging.info('start logging validation')
                                        # valid_writer.add_summary(valid_summary,
                                        #                          valid_iter_index + self.cur_epoch * VALID_ITERS)
                                        valid_writer.add_summary(valid_summary, gs)

                                        tf.logging.info(
                                            '[%s][Validation]epoch %s global_step %s, valid iter-%s'
                                            ' ave val_loss: %s, '
                                            'cur val_loss: %s' % (
                                                datetime.datetime.now(), last_epoch, gs, valid_iter_index,
                                                np.mean(valid_loss_list),
                                                np.mean(val_loss)
                                            )
                                            + '\tlabel 1 count: %s/%s, pred 1 count: %s/%s' % (
                                                len(filter(lambda x: x > 0.5, valid_label_batch)),
                                                len(valid_label_batch),
                                                len(filter(lambda x: x > 0.5, preds)), len(preds)))

                                except tf.errors.OutOfRangeError:
                                    tf.logging.warn('valid done, reach valid iter limit')
                                except Exception, e:
                                    tf.logging.error('valid error, details=%s' % e)
                                gs = sess.run(self.global_step)
                                # valid_calc_auc, valid_auc_value = sess.run([calc_auc, auc_op],
                                #                                            feed_dict={
                                #                                                labels: np.array(epoch_valid_labels),
                                #                                                sim_score: np.array(
                                #                                                    epoch_valid_preds)})
                                # valid_keras_auc = auc(epoch_valid_labels, epoch_valid_preds)
                                # valid_sklearn_auc = roc_auc_score(
                                #     np.array([int(np.round(t_l)) for t_l in epoch_valid_labels]), epoch_valid_preds)
                                tf.logging.info(
                                    '[%s]Epoch %s global_step %s,  total valid ACC: %s,'
                                    ' total pred 1: %s, total label 1: %s' % (
                                        datetime.datetime.now(), self.cur_epoch, gs,
                                        np.mean(mse_list),
                                        len(filter(lambda x: x > 0.5, label_list)),
                                        len(filter(lambda x: x > 0.5, pred_list))))
                                valid_epoch_summary = sess.run(epoch_merge, feed_dict={epoch_loss: valid_loss_list })
                                valid_writer.add_summary(valid_epoch_summary, last_epoch)
                                # EARLY-STOPPING
                                if is_warmup:
                                    tf.logging.info(
                                        '[warm up] saving model to %s' % (
                                            FLAGS.output_dir + '/%s/%s-early-stop-epoch%s' % (
                                                train_tag, train_tag, self.cur_epoch)))
                                    self.saver.save(sess, FLAGS.output_dir + '/%s-%s/%s-%s-early-stop-epoch%s' % (
                                        train_tag, str_date, train_tag, str_date, self.cur_epoch))
                                    continue
                                cur_epoch_val_loss = np.mean(valid_loss_list)
                                # cur_epoch_val_auc = np.mean([
                                #     valid_sklearn_auc])
                                if min_epoch_valid_loss <= cur_epoch_val_loss:
                                    # if max_epoch_valid_auc >= cur_epoch_val_auc:
                                    tf.logging.info(
                                        '[early-stop-info]val loss did not improve for %s epoches, best is %s, cur is %s, best epoch is %s' % (
                                            int(patience_count), min_epoch_valid_loss, cur_epoch_val_loss, best_epoch))
                                    patience_count += 1
                                else:
                                    patience_count = 0
                                    # tf.logging.info(
                                    #     '[early-stop-info]val loss improve from %s to %s, saving to %s' % (
                                    #         min_epoch_valid_loss, cur_epoch_val_loss,
                                    #         FLAGS.output_dir + '/%s-%s/%s-%s-early-stop-epoch%s' % (
                                    #             train_tag, str_date, train_tag, str_date, self.cur_epoch)))
                                    tf.logging.info(
                                        '[early-stop-info]val loss improve from %s to %s, saving to %s' % (
                                            min_epoch_valid_loss, cur_epoch_val_loss,
                                            FLAGS.output_dir + '/%s-%s/%s-%s-early-stop-epoch%s' % (
                                                train_tag, str_date, train_tag, str_date, self.cur_epoch)))
                                    # min_epoch_valid_loss = cur_epoch_val_loss
                                    # max_epoch_valid_auc = cur_epoch_val_auc
                                    best_epoch = self.cur_epoch
                                    self.saver.save(sess, FLAGS.output_dir + '/%s-%s/%s-%s-early-stop-epoch%s' % (
                                        train_tag, str_date, train_tag, str_date, self.cur_epoch))
                                if patience_count > TOTAL_PATIENCE:
                                    self.cur_epoch = FLAGS.max_epoch  # early-stop
                                    tf.logging.info(
                                        '[early-stop-info]val loss did not improve for %s epoches, reach limit %s times!' % (
                                            patience_count, TOTAL_PATIENCE))
                            continue
                            # except Exception, e:
                            #     tf.logging.info(
                            #         'training interupt at epoch %s -- unknown error: %s' % (self.cur_epoch, e))
                            #     break
                            # except Exception, e:
                            #     tf.logging.info('training failed -- unknown error: %s' % e)
        tf.logging.info("Optimization Finished!")


def main(_):
    if FLAGS.job_name == "worker" and FLAGS.task_index == 0:
        # if tf.gfile.Exists(FLAGS.log_dir):
        #     tf.gfile.DeleteRecursively(FLAGS.log_dir)
        if not tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.MakeDirs(FLAGS.log_dir)
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    tf.logging.info("----start---")

    tf.app.run()
