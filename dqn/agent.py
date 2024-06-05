import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from random import sample
from typing import Tuple

import gym
import numpy as np
import torch
from gym.wrappers import AtariPreprocessing, FrameStack
from torch.optim import RMSprop
from tqdm import trange

from model import DQN
from utils import float01, huber, toInt
from stable_baselines3.common.buffers import ReplayBuffer


class Agent:
    def __init__(
        self,
        game: str,
        #  replay_buffer_capacity: int,
        replay_buf: ReplayBuffer,
        replay_start_size: int,
        batch_size: int,
        discount_factor: float,
        lr: float,
        device: str = "cuda:0",
        env_seed: int = 0,
        frame_buffer_size: int = 4,
        print_self=True,
    ):

        self.device = device
        self.discount_factor = discount_factor
        self.game = game
        self.batch_size = batch_size
        self.replay_buf = replay_buf
        # self.replay_buf = ReplayBuffer(capacity=replay_buffer_capacity)

        self.env = FrameStack(
            AtariPreprocessing(
                gym.make(self.game),
                # noop_max=0,
                # terminal_on_life_loss=True,
                scale_obs=False,
            ),
            num_stack=frame_buffer_size,
        )
        self.env.seed(env_seed)
        self.reset()

        self.n_action = self.env.action_space.n
        self.policy_net = DQN(self.n_action).to(self.device)
        self.target_net = DQN(self.n_action).to(self.device).eval()
        self.optimizer = RMSprop(
            self.policy_net.parameters(),
            alpha=0.95,
            # momentum=0.95,
            eps=0.01,
        )

        if print_self:
            print(self)
        self._fill_replay_buf(replay_start_size)

    def __repr__(self):
        return "\n".join(
            [
                "Agent:",
                f"Game: {self.game}",
                f"Device: {self.device}",
                f"Policy net: {self.policy_net}",
                f"Target net: {self.target_net}",
                f"Replay buf: {self.replay_buf}",
            ]
        )

    def _fill_replay_buf(self, replay_start_size):
        for _ in trange(replay_start_size, desc="Fill replay_buf randomly", leave=True):
            self.step(1.0)

    def reset(self):
        """Reset the end, pre-populate self.frame_buf and self.state"""
        self.state = self.env.reset()

    @torch.no_grad()
    def step(self, epsilon, clip_reward=True):
        """
        Choose an action based on current state and epsilon-greedy policy
        """
        # Choose action
        if random.random() <= epsilon:
            q_values = None
            action = self.env.action_space.sample()
        else:
            torch_state = (
                torch.tensor(
                    self.state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                / 255.0
            )
            q_values = self.policy_net(torch_state)
            action = int(q_values.argmax(dim=1).item())

        # Apply action
        next_state, reward, done, _ = self.env.step(action)
        if clip_reward:
            reward = max(-1.0, min(reward, 1.0))

        # Store into replay buffer
        self.replay_buf.append(
            (
                torch.tensor(np.array(self.state), dtype=torch.float32, device="cpu")
                / 255.0,
                action,
                reward,
                torch.tensor(np.array(next_state), dtype=torch.float32, device="cpu")
                / 255.0,
                done,
            )
        )

        # Advance to next state
        self.state = next_state
        if done:
            self.reset()

        return reward, q_values, done

    def q_update(self):
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = [
            x.to(self.device) for x in self.replay_buf.sample(self.batch_size)
        ]

        with torch.no_grad():
            y = torch.where(
                dones,
                rewards,
                rewards + self.discount_factor * self.target_net(next_states).max(1)[0],
            )

        predicted_values = (
            self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        )
        loss = huber(y, predicted_values, 2.0)
        loss.backward()
        self.optimizer.step()
        return (y - predicted_values).abs().mean()
