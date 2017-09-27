
# coding: utf-8

# In[1]:

from vizdoom import *

import math
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import itertools as it
import pickle
from time import time, sleep

from collections import namedtuple
from copy import deepcopy
from PIL import Image
from skimage import transform, io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


# In[2]:

# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 20
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (3, 90, 120)
episodes_to_watch = 10

model_savefile = "./sp_weights.dump"
# Configuration file path
config_file_path = '../ViZDoom/scenarios/defend_the_center.cfg'


# In[3]:

def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


# In[4]:

game = initialize_vizdoom(config_file_path)


# In[5]:

def preprocess(image):
    image = transform.resize(image, resolution)
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = torch.from_numpy(image)
    return image

def get_current_state():
    return preprocess(game.get_state().screen_buffer)


# In[6]:

actions = [list(a) for a in it.product([0, 1], repeat=game.get_available_buttons_size())]


# In[7]:

class Replay:
    curIndex = 0
    size = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.replay = []
    def add(self, s1, s2, action, reward):
        if s2 is None:
           s2 = torch.zeros(s1.size())
        if self.size == self.capacity:
            self.replay[self.curIndex] = (s1, s2, action, reward)
        else:
            self.replay.append((s1, s2, action, reward))
            self.size = min(self.size + 1, self.capacity)
        self.curIndex = (self.curIndex + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.replay, batch_size)
replay = Replay(replay_memory_size)


# In[8]:

replay.size


# In[9]:

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 8, kernel_size=11, stride=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(320, 128)
        self.fc2 = nn.Linear(128, len(actions))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
dqn = DQN()


# In[10]:


mse = nn.MSELoss()
optimizer = optim.SGD(dqn.parameters(), lr=learning_rate)

def get_q(input):
    input = Variable(input)
    return dqn(input)

def get_action(input):
    qs = get_q(input)
    val, ind = torch.max(qs, 1)
    return ind.data.numpy()[0]

def learn_replay():
    if batch_size <= replay.size:
        sample = zip(*replay.sample(batch_size))
        s1, s2, action, reward = sample
        
        s1 = torch.stack(s1)
        #s1 = s1.view(s1.size()[0], 3, s1.size()[1], s1.size()[2])
        
        action = np.array(action)
        q1 = get_q(s1)
        print q1
        q1 = q1.data.numpy()[np.arange(action.size), action]
        s2 = torch.stack(s2)
        #s2 = s2.view(s2.size()[0], 3, s2.size()[1], s2.size()[2])
        q2 = get_q(s2).data.numpy()
        q2 = np.max(q2, 1)
        #q_func = np.vectorize(lambda s:np.max(get_q(s.view(1, 1, resolution[0], resolution[1]))) if s else 0)
        #q2 = q_func(s2)
        
        y = reward + discount_factor * q2
        
        q1 = Variable(torch.from_numpy(q1), requires_grad=True)
        y = Variable(torch.from_numpy(y).float())
        loss = mse(q1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        


# In[11]:

def learn(eps):
    action = 0
    s1 = get_current_state()
    if random.random() < eps:
        action = random.randint(0, len(actions)-1)
        print 'rand'
    else:
        action = get_action(s1.view(1, 3, s1.size()[1], s1.size()[2]))
        print 'not rand'
    reward = game.make_action(actions[action])
    if actions[action][2]:
        reward -= .05
    print reward
    s2 = None if game.is_episode_finished() else get_current_state()
    replay.add(s1, s2, action, reward)
    learn_replay()


# In[12]:

NUM_EPOCHS = 20
EPS_START = .95
EPS_END = .1
EPS_CONST = NUM_EPOCHS * .1
EPS_DECAY = NUM_EPOCHS * .70

for i in range(NUM_EPOCHS):
    print "%d EPOCH" % i
    learning_steps = 0
    num_episodes = 0
    epsilon = 0
    if i <= EPS_CONST:
        epsilon = EPS_START
    elif i <= EPS_DECAY:
        epsilon = EPS_START - (i - EPS_CONST) / (EPS_DECAY - EPS_CONST) * (EPS_START - EPS_END)
    else:
        epsilon = EPS_END
    scores = []
    while learning_steps < learning_steps_per_epoch:
        if(game.is_episode_finished() or num_episodes == 0):
            scores.append(game.get_total_reward())
            game.new_episode()
            num_episodes += 1
        learn(epsilon)
        learning_steps += 1
        if learning_steps % 200 == 0:
            print "%d of %d" % (learning_steps, learning_steps_per_epoch)
    print("Epoch score: %d" % np.mean(scores))
    torch.save(dqn, model_savefile)

game.close()
print("======================================")
print("Training finished. It's time to watch!")

# Reinitialize the game with window visible
game.set_window_visible(True)
game.set_mode(Mode.ASYNC_PLAYER)
game.init()

for _ in range(episodes_to_watch):
    game.new_episode()
    while not game.is_episode_finished():
        state = get_current_state()
        state = state.view(1, 3, resolution[1], resolution[2])
        best_action_index = get_action(state)

        # Instead of make_action(a, frame_repeat) in order to make the animation smooth
        game.make_action(actions[best_action_index])
        #for _ in range(frame_repeat):
        #    game.advance_action()

    # Sleep between episodes
    sleep(1.0)
    score = game.get_total_reward()
    print("Total score: ", score)


# In[13]:

# Reinitialize the game with window visible
game.set_window_visible(True)
game.set_mode(Mode.ASYNC_PLAYER)
game.init()

for _ in range(episodes_to_watch):
    game.new_episode()
    while not game.is_episode_finished():
        state = get_current_state()
        state = state.view(1, 3, resolution[1], resolution[2])
        best_action_index = get_action(state)

        # Instead of make_action(a, frame_repeat) in order to make the animation smooth
        game.make_action(actions[best_action_index])
        #for _ in range(frame_repeat):
        #    game.advance_action()

    # Sleep between episodes
    sleep(1.0)
    score = game.get_total_reward()
    print("Total score: ", score)

