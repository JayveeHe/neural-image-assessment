# coding=utf-8

"""
Created by jayveehe on 2019/2/12.
http://code.dianpingoa.com/hejiawei03
"""
import cPickle
import keras
import tensorflow as tf
from keras import Input
from keras.applications import ResNet50
from tensorflow.python.ops.image_ops_impl import ResizeMethod


def build_network(pkl_path=None):
    # word part
    # input_wordvec = tf.placeholder(dtype=tf.float32, shape=(None, 300), name='input_wordvec')
    # pic part
    input_pic = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3), name='input_pic')
    ## init resnet and weights
    resnet_model = ResNet50(include_top=False,
                            weights=None,
                            pooling='max')
    # if pkl_path:
    #     print 'loading model from %s' % pkl_path
    #     wfin = tf.gfile.GFile(pkl_path, 'rb')
    #     wlist = cPickle.load(wfin)
    #     resnet_model.set_weights(wlist)
    #     wfin.close()
    for layer in resnet_model.layers:
        # print layer.name
        layer.trainable = False
    # resize input pic
    resized_input_pic = tf.image.resize_images(input_pic, size=(224, 224), method=ResizeMethod.NEAREST_NEIGHBOR)
    # imagenet preprocess
    processed_img = resized_input_pic[..., ::-1]
    processed_img = tf.subtract(processed_img, [[103.939, 116.779, 123.68]], name='input_processed_img')

    resnet_output = resnet_model(processed_img)

    # concat pic and wordvec
    with tf.variable_scope('querymatch_trainable'):
        # logits = keras.layers.
        # fc256 = keras.layers.Dense(256, activation='relu', name='FC256')(resnet_output)
        # bn_1 = keras.layers.BatchNormalization()(fc256)
        # fc300 = keras.layers.Dense(300, activation='relu', name='fc300')(bn_1)
        # # bn2=keras.layers.BatchNormalization()(fc64)
        # concat_word_pic = keras.layers.concatenate([input_wordvec, fc300], name='concat_word_pic')
        logits = keras.layers.Dense(1, activation='softmax', name='sim_score')(resnet_output)
    trainable_vars_list = tf.trainable_variables('querymatch_model/querymatch_trainable')
    return input_pic, logits, trainable_vars_list, resnet_model


if __name__ == '__main__':
    input_wordvec, input_pic, fc_300, sim_score, trainable_vars_list_1, resnet_model = build_network()
    print len(trainable_vars_list_1)
