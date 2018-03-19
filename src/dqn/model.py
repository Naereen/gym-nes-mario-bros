#!/usr/bin/env python
# -*- coding: utf-8 -*-
# By Lilian Besson (Naereen)
# https://github.com/Naereen/gym-nes-mario-bros
# MIT License https://lbesson.mit-license.org/
#
from __future__ import division, print_function  # Python 2 compatibility

import os
from collections import deque
import numpy as np

import keras
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.layers import Dense, Input, MaxPooling2D
from keras.models import Model, load_model, save_model
from keras import optimizers
# from keras.callbacks import TensorBoard
from keras import backend as K

from .replay_buffer import ReplayBuffer
from .utils import LinearSchedule, PiecewiseSchedule


# DEBUG
debug_print = True
debug_print = False

# WARNING try both! Not sure what it does... No doc!
padding = 'valid'
padding = 'same'


def q_function(input_shape, num_actions):
    """Description of the Q-function as Keras model."""
    # See also https://github.com/PacktPublishing/Practical-Deep-Reinforcement-Learning/blob/2e16284bb7661a7edd908cefc5fe2cfb55ac57d8/ch07/lib/dqn_model.py#L64
    # https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py
    # https://github.com/yunjhongwu/Double-DQN-Breakout/blob/master/Player.py#L17
    # FIXME find the best possible architecture!
    image_input = Input(shape=input_shape)

    # https://keras.io/layers/convolutional/#conv2d
    out = Conv2D(filters=32, kernel_size=8, strides=(4, 4), padding=padding, activation='relu')(image_input)

    out = Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding=padding, activation='relu')(out)
    # out = Conv2D(filters=32, kernel_size=4, strides=(2, 2), padding=padding, activation='relu')(out)

    out = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding=padding, activation='relu')(out)
    # out = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding=padding, activation='relu')(out)
    # out = Conv2D(filters=64, kernel_size=2, strides=(1, 1), padding=padding, activation='relu')(out)

    # WARNING not sure!
    out = MaxPooling2D(pool_size=(2, 2))(out)

    out = Flatten()(out)

    # out = Dense(512, activation='relu')(out)
    # out = Dense(256, activation='relu')(out)
    out = Dense(128, activation='relu')(out)
    out = Dense(128, activation='relu')(out)

    q_value = Dense(num_actions)(out)

    return image_input, q_value


def q_model(input_shape, num_actions):
    inputs, outputs = q_function(input_shape, num_actions)
    return Model(inputs, outputs)


# XXX ValueError: probabilities are not non-negative
SAMPLE_FROM_Q_VALS = False


class DoubleDQN(object):
    def __init__(self,
                image_shape,
                num_actions,
                frame_history_len=4,
                replay_buffer_size=10000,
                training_freq=4,
                training_starts=5000,
                training_batch_size=32,
                target_update_freq=1000,
                reward_decay=0.99,
                exploration=None,
                sample_from_q_vals=SAMPLE_FROM_Q_VALS,
                log_dir="logs/",
                name="DQN",
                optimizer=None
            ):
        """ Double Deep Q Network

            Parameters
            ----------
            image_shape: (height, width, n_values)
            num_actions: how many different actions we can choose
            frame_history_len: feed this number of frame data as input to the deep-q Network
            replay_buffer_size: size limit of replay buffer
            training_freq: train base q network once per training_freq steps
            training_starts: only train q network after this number of steps
            training_batch_size: batch size for training base q network with gradient descent
            reward_decay: decay factor(called gamma in paper) of rewards that happen in the future
            exploration: used to generate an exploration factor(see 'epsilon-greedy' in paper). When rand(0,1) < epsilon, take random action; otherwise take greedy action.
            sample_from_q_vals: True to sample from the distribution of action, False to always take the most likely (sample or argmax).
            log_dir: path to write tensorboard logs
        """
        super().__init__()
        self.name                = name
        self.num_actions         = num_actions
        self.training_freq       = training_freq
        self.training_starts     = training_starts
        self.training_batch_size = training_batch_size
        self.target_update_freq  = target_update_freq
        self.reward_decay        = reward_decay
        if exploration is None:
            exploration = LinearSchedule(5000, 0.1)
        self.exploration         = exploration
        self.sample_from_q_vals  = sample_from_q_vals

        # use multiple frames as input to q network
        input_shape = image_shape[:-1] + (image_shape[-1] * frame_history_len,)
        # used to choose action
        self.base_model = q_model(input_shape, num_actions)
        if optimizer is None:
            optimizer = optimizers.adam(clipnorm=10, lr=1e-4, decay=1e-6, epsilon=1e-4)
        self.base_model.compile(optimizer=optimizer, loss='mse')
        # used to estimate q values
        self.target_model = q_model(input_shape, num_actions)

        self.replay_buffer = ReplayBuffer(size=replay_buffer_size, frame_history_len=frame_history_len)
        # current replay buffer offset
        self.replay_buffer_idx = 0

        # self.tensorboard_callback = TensorBoard(log_dir=log_dir)
        self.latest_losses = deque(maxlen=100)

    # --- Interface to Keras models

    def summary(self):
        print("Summary of the model:")
        self.target_model.summary()

    def plot_model(self, base_model_path=None, target_model_path=None):
        # https://keras.io/utils/#plot_model
        if base_model_path is None: base_model_path = "dqn_base.svg"
        keras.utils.plot_model(self.base_model, to_file=base_model_path, show_shapes=True, show_layer_names=True)
        if target_model_path is None: target_model_path = "dqn_target.svg"
        keras.utils.plot_model(self.target_model, to_file=target_model_path, show_shapes=True, show_layer_names=True)

    # https://keras.io/models/about-keras-models/

    def load_weights(self, base_model_path=None, target_model_path=None):
        if base_model_path is None: base_model_path = self.name + "_base.h5"
        self.base_model.load_weights(base_model_path)
        if target_model_path is None: target_model_path = self.name + "_target.h5"
        self.target_model.load_weights(target_model_path)

    def save_weights(self, base_model_path=None, target_model_path=None, overwrite=True):
        if base_model_path is None: base_model_path = self.name + "_base.h5"
        self.base_model.save_weights(base_model_path, overwrite=overwrite)
        if target_model_path is None: target_model_path = self.name + "_target.h5"
        self.target_model.save_weights(target_model_path, overwrite=overwrite)

    def save_model(self, base_model_path=None, target_model_path=None, yaml=False, overwrite=True):
        extension = (".yaml" if yaml else ".json")
        if base_model_path is None: base_model_path = self.name + "_base" + extension
        if target_model_path is None: target_model_path = self.name + "_target" + extension

        for model, path in [
                (self.base_model, base_model_path),
                (self.target_model, target_model_path),
            ]:
            method = model.to_yaml if yaml else model.to_json
            # don't overwrite existing file
            if os.path.isfile(path) and not overwrite:
                print("save_model failed, as the file {} already existed (you can force to overwrite it with 'overwrite=True'...".format(path))  # DEBUG
                return 1
            model_string = method()
            with open(path, 'w') as f:
                return f.write(model_string)

    # ---

    def choose_action(self, step, obs):
        self.replay_buffer_idx = self.replay_buffer.store_frame(obs)
        if step < self.training_starts or np.random.rand() < self.exploration.value(step):
            # take random action
            action = np.random.randint(self.num_actions)
        else:
            # take action that results in maximum q value
            recent_obs = self.replay_buffer.encode_recent_observation()
            q_vals = self.base_model.predict_on_batch(np.array([recent_obs])).flatten()
            if self.sample_from_q_vals:
                action = np.random.choice(len(q_vals), p=q_vals/np.sum(q_vals))
            else:
                action = np.argmax(q_vals)
        return action

    def learn(self, step, action, reward, done, info=None):
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)
        if step > self.training_starts and step % self.training_freq == 0:
            self._train()

        if step > self.training_starts and step % self.target_update_freq == 0:
            self._update_target()

    def get_learning_rate(self):
        optimizer = self.base_model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * K.cast(optimizer.iterations, K.tf.float32))))
        return lr

    def get_avg_loss(self):
        if len(self.latest_losses) > 0:
            return np.mean(np.array(self.latest_losses))
        else:
            return None

    def _train(self):
        obs_t, action, reward, obs_t1, done_mask = self.replay_buffer.sample(self.training_batch_size)
        q = self.base_model.predict(obs_t)
        q_t1 = self.target_model.predict(obs_t1)
        q_t1_max = np.max(q_t1, axis=1)
        if debug_print:
            print("q:\n", q)  # DEBUG
            print("q_t1:\n", q_t1)  # DEBUG
            print("q_t1_max:\n", q_t1_max)  # DEBUG
            print("action:\n", action)  # DEBUG

        q[range(len(action)), action] = reward + q_t1_max * self.reward_decay * (1-done_mask)

        if debug_print:
            print("reward:\n", reward)  # DEBUG
            print("qt1_max:\n", q_t1_max)  # DEBUG
            print("done mask:\n", done_mask)  # DEBUG
            print("q: \n", q)  # DEBUG

        # FIXME How to enable the TensorBoard callback?
        # self.base_model.fit(obs_t, q, batch_size=self.training_batch_size, epochs=1, callbacks=[self.tensorboard_callback])

        loss = self.base_model.train_on_batch(obs_t, q)
        self.latest_losses.append(loss)

    def _update_target(self):
        weights = self.base_model.get_weights()
        if debug_print:
            print("update target:\n", weights)  # DEBUG
        self.target_model.set_weights(weights)
