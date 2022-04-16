"""Model specification for SimCLR: Cutoff VGG16"""

from absl import flags
import tensorflow.compat.v2 as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

FLAGS = flags.FLAGS


class CutoffVGG16(tf.keras.layers.Layer):

    def __init__(self, dropout_rate, cutoff_layer=10, init_weights='imagenet', data_format='channels_last', **kwargs):
        super(CutoffVGG16, self).__init__(**kwargs)
        self.data_format = data_format
        self.dropout_rate = dropout_rate
        self.cutoff_layer = cutoff_layer

        vgg16 = VGG16(input_shape=self.input_shape, include_top=False, weights=init_weights)
        self.vgg16_layers = []
        layer_idx = 0
        for layer in vgg16.layers[1:self.cutoff_layer]:
            trainable = (FLAGS.train_mode != 'finetune' or FLAGS.fine_tune_after_block == layer_idx)
            layer.trainable = trainable
            self.vgg16_layers.append(layer)
            layer_idx += 1
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dropout = Dropout(self.dropout_rate)
        trainable = (FLAGS.train_mode != 'finetune' or FLAGS.fine_tune_after_block == self.cutoff_layer + 2)
        self.output_layer = Dense(1, activation='sigmoid', trainable=trainable)

    def call(self, inputs, training):
        x = inputs
        for layer in self.vgg16_layers:
            x = layer(x, training=training)
        x = self.global_avg_pool(x)
        x = self.dropout(x)
        return self.output_layer(x)
