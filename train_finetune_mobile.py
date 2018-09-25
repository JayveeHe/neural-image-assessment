import os

from keras.applications import MobileNet
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras import backend as K
import keras

from utils.dianping_dataset_utils import train_generator, val_generator

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
'''
Below is a modification to the TensorBoard callback to perform 
batchwise writing to the tensorboard, instead of only at the end
of the batch.
'''


class TensorBoardBatch(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(TensorBoardBatch, self).__init__(*args, **kwargs)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, batch)

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch * self.batch_size)

        self.writer.flush()


def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)


image_size = 224

base_model = MobileNet((image_size, image_size, 3), alpha=1, include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = False

x = Dropout(0.75)(base_model.output)
x = Dense(10, activation='softmax')(x)

model = Model(base_model.input, x)
model.summary()
optimizer = Adam(lr=1e-3)
model.compile(optimizer, loss=earth_mover_loss)

# load weights from trained model if it exists
if os.path.exists('weights/mobilenet_weights.h5'):
    model.load_weights('weights/mobilenet_weights.h5')

# load pre-trained NIMA(NASNet Mobile) classifier weights
# if os.path.exists('weights/nasnet_pretrained_weights.h5'):
#     model.load_weights('weights/nasnet_pretrained_weights.h5', by_name=True)

checkpoint = ModelCheckpoint('weights/finetune_mobilenet_weights.h5', monitor='val_loss', verbose=1,
                             save_weights_only=False, save_best_only=True,
                             mode='min')
early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
tensorboard = TensorBoardBatch(log_dir='./finetune_mobilenet_logs/')
callbacks = [checkpoint, tensorboard, early_stopper]

batchsize = 200
epochs = 20

model.fit_generator(train_generator(batchsize=batchsize),
                    steps_per_epoch=(250000. // batchsize),
                    epochs=epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_generator(batchsize=batchsize),
                    validation_steps=(5000. // batchsize))
