# implement different replay methods in this file
from collections import deque
import random
import torch
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Buffer:
    def __init__(self, size, batch_size):
        self.max_size = size
        self.current_size = 0
        self.data = deque(maxlen=size)
        self.batch_size = batch_size

    def add(self, state, action, reward, newstate, done):
        if self.current_size < self.max_size:
            self.data.append((state, int(action), reward, newstate, not done))
            # self.data.append((state.data.numpy(),action,reward,newstate,done))
            self.current_size += 1
        else:
            self.data.popleft()
            self.data.append((state, int(action), reward, newstate, not done))

    def get_batch(self):
        batch = random.sample(self.data, self.batch_size)
        batch = np.array(batch)
        state = Variable(torch.cat(batch[:, 0]))
        action = torch.LongTensor(batch[:, 1])
        reward = Variable(torch.FloatTensor(batch[:, 2]))
        next_state = Variable(torch.cat(batch[:, 3]))
        not_done = Variable(torch.Tensor(batch[:, 4]).float())
        return state, action, reward, next_state, not_done


class Memory:
    def __init__(self, len):
        self.rewards = deque(maxlen=len)
        self.state = deque(maxlen=len)
        self.action = deque(maxlen=len)
        self.is_done = deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.is_done)
        idx = random.sample(range(0, n - 1), batch_size)

        return (
            torch.Tensor(self.state)[idx].to(device),
            torch.LongTensor(self.action)[idx].to(device),
            torch.Tensor(self.state)[1 + np.array(idx)].to(device),
            torch.Tensor(self.rewards)[idx].to(device),
            torch.Tensor(self.is_done)[idx].to(device),
        )

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()
