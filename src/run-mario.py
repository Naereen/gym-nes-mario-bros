#!/usr/bin/env python
# -*- coding: utf-8 -*-
# By Lilian Besson (Naereen)
# https://github.com/Naereen/gym-nes-mario-bros
# MIT License https://lbesson.mit-license.org/
#
from __future__ import division, print_function  # Python 2 compatibility

import os
from collections import deque

import gym
from gym import wrappers
import nesgym
import numpy as np

from dqn.model import DoubleDQN
from dqn.utils import PiecewiseSchedule


def get_env():
    env = gym.make('nesgym/MarioBros-v0')
    env = nesgym.wrap_nes_env(env)
    expt_dir = '/tmp/mario/'
    env = wrappers.Monitor(env, os.path.join(expt_dir, "gym"), force=True)
    return env


def mario_main():
    env = get_env()

    last_obs = env.reset()

    max_timesteps = 400000
    max_seen_score = 0

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e5, 0.1),
            (max_timesteps / 2, 0.01),
        ], outside_value=0.01
    )

    dqn = DoubleDQN(image_shape=(84, 84, 1),
                    num_actions=env.action_space.n,
                    # XXX Heavy simulations
                    # training_starts=10000,
                    # target_update_freq=4000,
                    # training_batch_size=64,
                    # XXX light simulations?
                    training_starts=5000,
                    target_update_freq=100,
                    training_batch_size=4,
                    # Other parameters
                    frame_history_len=4,
                    replay_buffer_size=100000,  # XXX reduce if MemoryError
                    exploration=exploration_schedule
                )

    reward_sum_episode = 0
    num_episodes = 0
    episode_rewards = deque(maxlen=100)

    for step in range(max_timesteps):
        if step > 0 and step % 100 == 0:
            print("step: ", step,
                  "; episodes:", num_episodes,
                  "; epsilon:", exploration_schedule.value(step),
                  "; learning rate:", dqn.get_learning_rate(),
                  "; last 100 training loss mean", dqn.get_avg_loss()
            )
            if len(episode_rewards) > 0:
                print("last 100 episode mean rewards: ", np.mean(np.array(episode_rewards)))

        # XXX Enable this to see the Python view of the screen
        # env.render()

        action = dqn.choose_action(step, last_obs)
        obs, reward, done, info = env.step(action)
        reward_sum_episode += reward
        if done and reward < 0:
            reward = 0  # force this manually!
        dqn.learn(step, action, reward, done, info)
        print("Step", step, "\t action =", action, "gave reward =", reward, " score =", info['score'], "and max score =", max_seen_score)  # DEBUG

        if info['score'] > max_seen_score:
            max_seen_score = info['score']
            print("New score record!!", max_seen_score)
        if done:
            last_obs = env.reset()
            if info['frame'] > 0:  # we actually played a few frames
                print("\ndone, reward_sum_episode =", reward_sum_episode)
                episode_rewards.append(reward_sum_episode)
                reward_sum_episode = 0
                num_episodes += 1
        else:
            last_obs = obs

        last_obs = obs

if __name__ == "__main__":
    mario_main()
