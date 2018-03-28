#!/usr/bin/env python
# -*- coding: utf-8 -*-
# By Lilian Besson (Naereen)
# https://github.com/Naereen/gym-nes-mario-bros
# MIT License https://lbesson.mit-license.org/
#
from __future__ import division, print_function  # Python 2 compatibility

import gym
import numpy as np

from dqn.model import DoubleDQN
from dqn.atari_wrappers import wrap_deepmind
from dqn.utils import PiecewiseSchedule

from collections import deque

def get_env(task, seed):
    env_id = task.env_id
    env = gym.make(env_id)
    env.seed(seed)
    env = wrap_deepmind(env)
    return env


def atari_main(env_id='Pong-v0'):
    # Run training
    max_timesteps = 100000
    print('task: ', env_id, 'max steps: ', max_timesteps)
    env = gym.make(env_id)

    last_obs = env.reset()

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (max_timesteps / 2, 0.01),
        ], outside_value=0.01
    )

    dqn = DoubleDQN(
                    # image_shape=(84, 84, 1),
                    image_shape=(210, 160, 3),  # FIXME debug this!
                    num_actions=env.action_space.n,
                    # # --- XXX heavy simulations
                    training_starts=10000,
                    target_update_freq=1000,
                    training_batch_size=32,
                    training_freq=4,
                    # # --- XXX light simulations?
                    # training_starts=1000,
                    # target_update_freq=100,
                    # training_batch_size=4,
                    # training_freq=4,
                    # --- Other parameters...
                    # frame_history_len=1,  # XXX is it more efficient with history?
                    # replay_buffer_size=10000,  # XXX reduce if MemoryError
                    frame_history_len=8,  # XXX is it more efficient with history?
                    replay_buffer_size=100000,  # XXX reduce if MemoryError
                    exploration=exploration_schedule
                )

    dqn.summary()

    reward_sum_episode = 0
    num_episodes = 0
    episode_rewards = deque(maxlen=100)

    for step in range(max_timesteps):
        if step > 0 and step % 1000 == 0:
            print('step: ', step, 'episodes:', num_episodes, 'epsilon:', exploration_schedule.value(step),
                  'learning rate:', dqn.get_learning_rate(), 'last 100 training loss mean', dqn.get_avg_loss(),
                  'last 100 episode mean rewards: ', np.mean(np.array(episode_rewards, dtype=np.float32)))
        # if step > 0 and step % 100 == 0:
        #     dqn.summary()

        env.render()

        action = dqn.choose_action(step, last_obs)
        obs, reward, done, info = env.step(action)
        reward_sum_episode += reward
        dqn.learn(step, action, reward, done, info)

        # print("Step {:>6}, action #{:>2}, gave reward {:>6}.".format(step, action, reward))  # DEBUG

        if done:
            last_obs = env.reset()
            episode_rewards.append(reward_sum_episode)
            reward_sum_episode = 0
            num_episodes += 1
        else:
            last_obs = obs


if __name__ == "__main__":
    import sys
    env_id = 'Pong-v0'  # --> OK the DQN model works!

    # https://gym.openai.com/envs/#atari
    if any('pong' in arg for arg in sys.argv):
        env_id = 'Pong-v0'  # --> OK the DQN model works!

    if any('breakout' in arg for arg in sys.argv):
        env_id = 'Breakout-v0'

    if any('pacman' in arg for arg in sys.argv):
        env_id = 'MsPacman-v0'

    if any('invaders' in arg for arg in sys.argv):
        env_id = 'SpaceInvaders-v0'

    atari_main(env_id=env_id)
