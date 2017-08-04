import keras
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.layers import Dense, Input
from keras.models import Model
from keras import optimizers

import numpy as np

from .replay_buffer import ReplayBuffer


def q_function(input_shape, num_actions):
    image_input = Input(shape=input_shape)
    out = Conv2D(filters=32, kernel_size=8, strides=(4, 4), padding='valid', activation='relu')(image_input)
    out = Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='valid', activation='relu')(out)
    out = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', activation='relu')(out)
    out = Flatten()(out)
    out = Dense(512, activation='relu')(out)
    q_value = Dense(num_actions)(out)

    return image_input, q_value


def q_model(input_shape, num_actions):
    inputs, outputs = q_function(input_shape, num_actions)
    return Model(inputs, outputs)


class DoubleDQN(object):
    def __init__(self,
                 image_shape,
                 num_actions):
        super().__init__()
        self.num_actions = num_actions

        # use multiple frames as input to q network
        frame_history_len = 4
        input_shape = image_shape[:-1] + (image_shape[-1] * frame_history_len,)
        # used to choose action
        self.base_model = q_model(input_shape, num_actions)
        self.base_model.compile(optimizer=optimizers.rmsprop(clipnorm=10), loss='mse')
        # used to estimate q values
        self.target_model = q_model(input_shape, num_actions)

        self.replay_buffer = ReplayBuffer(size=1000000, frame_history_len=frame_history_len)

    def choose_action(self, step, obs):
        if step < 20:
            # take random action
            action = np.random.randint(self.num_actions)
        else:
            # take action that results in maximum q value
            obs = self.replay_buffer.encode_recent_observation()
            q_vals = self.base_model.predict_on_batch(np.array([obs])).flatten()
            action = np.argmax(q_vals)
        return action

    def learn(self, step, obs, action, reward, done, info=None):
        idx = self.replay_buffer.store_frame(obs)
        self.replay_buffer.store_effect(idx, action, reward, done)
        learning_freq = 4
        learning_starts = 10
        if step > learning_starts and step % learning_freq == 0:
            self._train()

        target_update_freq = 10
        if step > learning_starts and step % target_update_freq == 0:
            self._update_target()

    def _train(self):
        # batch_size = 32
        batch_size = 2
        obs_t, action, reward, obs_t1, done_mask = self.replay_buffer.sample(batch_size)
        q = self.base_model.predict(obs_t)
        q_t1_max = np.max(self.target_model.predict(obs_t1), axis=1)
        gamma = 0.99
        for idx in range(len(q)):
            q[idx][action] = reward[idx] + q_t1_max[idx] * gamma * done_mask[idx]
        self.base_model.train_on_batch(obs_t, q)

    def _update_target(self):
        weights = self.base_model.get_weights()
        self.target_model.set_weights(weights)
