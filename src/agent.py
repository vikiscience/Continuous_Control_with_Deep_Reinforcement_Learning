# -*- coding: utf-8 -*-
import const
from src import models, buffer

from pathlib import Path
import numpy as np
import torch

torch.random.manual_seed(const.random_seed)  # todo


class DRLAgent:
    model_path = const.file_path_model

    def __init__(self, num_states: int = const.state_size,
                 num_actions: int = const.action_size,
                 memory_size: int = const.memory_size,
                 gamma: float = const.gamma,
                 batch_size: int = const.batch_size,
                 expl_noise: int = const.expl_noise,
                 model_learning_rate: float = const.model_learning_rate,
                 model_fc1_num: int = const.model_fc1_num,
                 model_fc2_num: int = const.model_fc2_num
                 ):

        # agent params
        self.num_states = num_states
        self.num_actions = num_actions
        self.memory = buffer.ReplayBuffer(memory_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.expl_noise = expl_noise
        self.start_policy_training_iter = batch_size  # start training after: buffer_size >= batch_size

        # model params
        self.model_learning_rate = model_learning_rate
        self.model_fc1_num = model_fc1_num
        self.model_fc2_num = model_fc2_num

        self.policy = models.TD3(state_dim=self.num_states, action_dim=self.num_actions,
                                 max_action=const.max_action, discount=self.gamma)

    def act(self, states, t):
        # Select action randomly or according to policy
        if t < self.start_policy_training_iter:
            action = np.random.randn(const.num_agents, self.num_actions)
        else:
            action = (
                    self.policy.select_action(np.array(states[0]))  # todo [0]
                    + np.random.normal(0, const.max_action * self.expl_noise, size=self.num_actions)
            )
        actions = action.reshape(1, -1)
        actions = np.clip(actions, - const.max_action, const.max_action)  # all actions between -1 and 1

        return actions

    def do_stuff(self, state, action, reward, next_state, done, t):
        self.memory.append((state, action, reward, next_state, done))  # memorize

        # Train agent after collecting sufficient data
        if t >= self.start_policy_training_iter:
            self.policy.train(self.memory, self.batch_size)

    def load(self):
        const.myprint('Loading model from:', self.model_path)
        self.policy.load(str(self.model_path))

    def save(self):
        const.myprint('Saving model to:', self.model_path)
        self.policy.save(str(self.model_path))

    def set_model_path(self, i):
        p = self.model_path
        self.model_path = Path(p.parent, 'model_' + str(i) + p.suffix)
