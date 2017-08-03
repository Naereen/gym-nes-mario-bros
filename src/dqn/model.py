import keras
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.layers import Dense, Input
from keras.models import Model


def q_function(image_shape, num_actions):
    image_input = Input(shape=image_shape)
    out = Conv2D(filters=32, kernel_size=8, strides=(4, 4), padding='valid', activation='relu')(image_input)
    out = Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='valid', activation='relu')(out)
    out = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', activation='relu')(out)
    out = Flatten()(out)
    out = Dense(512, activation='relu')(out)
    q_value = Dense(num_actions)(out)

    return image_input, q_value


def q_model(image_shape, num_actions):
    inputs, outputs = q_function(image_shape, num_actions)
    return Model(inputs, outputs)
