import keras
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.layers import Dense, Input
from keras.models import Model
from keras import optimizers
from keras.callbacks import TensorBoard

import numpy as np

from .replay_buffer import ReplayBuffer
from .utils import LinearSchedule, PiecewiseSchedule


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
                 num_actions,
                 frame_history_len=4,
                 replay_buffer_size=1000000,
                 training_freq=4,
                 training_starts=5000,
                 training_batch_size=32,
                 target_update_freq=1000,
                 reward_decay=0.99,
                 exploration=LinearSchedule(5000, 0.1),
                 log_dir="logs/"):
        """
            Double Deep Q Network
            params:
            image_shape: (height, width, n_values)
            num_actions: how many different actions we can choose
            frame_history_len: feed this number of frame data as input to the deep-q Network
            replay_buffer_size: size limit of replay buffer
            training_freq: train base q network once per training_freq steps
            training_starts: only train q network after this number of steps
            training_batch_size: batch size for training base q network with gradient descent
            reward_decay: decay factor(called gamma in paper) of rewards that happen in the future
            exploration: used to generate an exploration factor(see 'epsilon-greedy' in paper).
                         when rand(0,1) < epsilon, take random action; otherwise take greedy action.
            log_dir: path to write tensorboard logs
        """
        super().__init__()
        self.num_actions = num_actions
        self.training_freq = training_freq
        self.training_starts = training_starts
        self.training_batch_size = training_batch_size
        self.target_update_freq = target_update_freq
        self.reward_decay = reward_decay
        self.exploration = exploration

        # use multiple frames as input to q network
        input_shape = image_shape[:-1] + (image_shape[-1] * frame_history_len,)
        # used to choose action
        self.base_model = q_model(input_shape, num_actions)
        self.base_model.compile(optimizer=optimizers.rmsprop(clipnorm=10), loss='mse')
        # used to estimate q values
        self.target_model = q_model(input_shape, num_actions)

        self.replay_buffer = ReplayBuffer(size=replay_buffer_size, frame_history_len=frame_history_len)

        self.tensorboard_callback = TensorBoard(log_dir=log_dir)

    def choose_action(self, step, obs):
        if step < self.training_starts or np.random.rand() < self.exploration.value(step):
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
        if step > self.training_starts and step % self.training_freq == 0:
            self._train()

        if step > self.training_starts and step % self.target_update_freq == 0:
            self._update_target()

    def _train(self):
        obs_t, action, reward, obs_t1, done_mask = self.replay_buffer.sample(self.training_batch_size)
        q = self.base_model.predict(obs_t)
        q_t1_max = np.max(self.target_model.predict(obs_t1), axis=1)
        for idx in range(len(q)):
            q[idx][action] = reward[idx] + q_t1_max[idx] * self.reward_decay * (1-done_mask[idx])
        # self.base_model.fit(obs_t, q, batch_size=self.training_batch_size, epochs=1)
        self.base_model.train_on_batch(obs_t, q)

    def _update_target(self):
        weights = self.base_model.get_weights()
        self.target_model.set_weights(weights)
