# coding=utf-8

"""
Created by jayveehe on 2019/2/12.
http://code.dianpingoa.com/hejiawei03
"""

import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from PIL import Image
import StringIO

from w2v_utils import Word2VecManager


def read_from_tfrecord(filepath_list, batchsize,w2v_manager):
    def _parse_feature(data):
        with tf.name_scope('tfrecord_inputs'):
            features = tf.parse_single_example(data,
                                               features={
                                                   'merge_queryid': tf.FixedLenFeature([], tf.string),
                                                   'keyword': tf.FixedLenFeature([], tf.string),
                                                   'newpicurl': tf.FixedLenFeature([], tf.string),
                                                   'pickey': tf.FixedLenFeature([], tf.string),
                                                   'picurl': tf.FixedLenFeature([], tf.string),
                                                   'flag': tf.FixedLenFeature([1], tf.int64),
                                                   'picdata': tf.FixedLenFeature([], tf.string)
                                               })
            img_1 = tf.image.decode_image(features['picdata'], channels=3, name='input_raw_picdata')
            img_1.set_shape([None, None, None])
            img_1 = tf.cast(img_1, dtype=tf.int8)
            resized_img_1 = tf.image.resize_images(img_1, size=(224, 224), method=ResizeMethod.NEAREST_NEIGHBOR)
            keyword = features['keyword']
            kw_vec = w2v_manager.lookup_wordvec(keyword)
            flag = features['flag']
            picurl = features['picurl']

            return resized_img_1, keyword,kw_vec, flag, picurl

            # return img_1, a_picid, a_totalscore, img_2, b_picid, b_totalscore, final_score

    # fnamelist = tf.gfile.ListDirectory(root_path)
    # pathlist = [os.path.join(root_path, fname) for fname in fnamelist]
    tfdata = tf.data.TFRecordDataset(filepath_list)
    parsed_data = tfdata.map(_parse_feature, num_parallel_calls=5)
    prefetch_data = parsed_data.prefetch(buffer_size=batchsize * 5)
    shuffled_data = prefetch_data.shuffle(buffer_size=batchsize * 2)
    batch_data = shuffled_data.batch(batchsize)
    return batch_data


if __name__ == '__main__':
    w2v_m = Word2VecManager('/Users/jayveehe/Jobs/project-pikachu/data/tmp_dict', 300)
    train_dataset = read_from_tfrecord(['/Users/jayveehe/Jobs/project-pikachu/data/tfrecords/part-r-00099'],
                                       batchsize=5, w2v_manager=w2v_m)
    common_dataset_iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                          train_dataset.output_shapes)
    img_obj, keyword,keyword_vec, flag, picurl = common_dataset_iter.get_next()
    # labels = label_batch
    train_initializer = common_dataset_iter.make_initializer(train_dataset)
    with tf.Session() as sess:
        sess.run(train_initializer)
        img, kw,kv, fg, pu = sess.run([img_obj, keyword,keyword_vec, flag, picurl])
        for s in img:
            Image.fromarray(s, 'RGB').show()
        print 'test'
